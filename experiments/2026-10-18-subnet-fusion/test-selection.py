from pathlib import Path
import subprocess
import sys
import tempfile

source = Path(sys.argv[1]).read_text()
helpers = source[source.index('static bool gidSameSubnet('):source.index('\nncclResult_t ncclIbListen(')]
harness = r'''
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <cassert>
#include <cstring>
#include <cstdio>
using ncclResult_t = int;
constexpr int ncclSuccess=0, ncclInvalidArgument=1, ncclInvalidUsage=2;
#define INFO(...) do {} while(0)
#define WARN(...) do {} while(0)
#define NCCLCHECK(call) do { int r=(call); if (r) return r; } while(0)
struct Props { int ndevs; int devs[4]; };
struct ncclIbMergedDev { Props vProps; } ncclIbMergedDevs[6];
struct Port { int link_layer=IBV_LINK_LAYER_ETHERNET; };
struct ncclIbDev { int context; int portNum=1; Port portAttr; } ncclIbDevs[4];
int ncclNIbDevs=4, ncclNMergedIbDevs=6, prefix=24, failed=-1;
union ibv_gid mockGids[4];
int ncclParamIbSubnetPrefixLen() { return prefix; }
int getGidAddrFamily(union ibv_gid*) { return AF_INET; }
bool validGid(union ibv_gid* g) { return g->global.interface_id!=0; }
int ncclIbGetGidIndex(int context,int,Port*,int* index) { *index=3; return context==failed; }
int wrap_ibv_query_gid(int context,int,int,union ibv_gid* g) { *g=mockGids[context]; return 0; }
union ibv_gid gid(const char* ip) {
 union ibv_gid g={}; g.raw[10]=255; g.raw[11]=255;
 assert(inet_pton(AF_INET,ip,g.raw+12)==1); return g;
}
'''
cases = r'''
int main() {
 const char* ips[4]={"10.100.224.1","10.100.224.3","10.100.227.2","10.100.227.4"};
 for (int i=0;i<4;i++) { ncclIbDevs[i].context=i; mockGids[i]=gid(ips[i]); ncclIbMergedDevs[i].vProps={1,{i}}; }
 ncclIbMergedDevs[4].vProps={2,{0,1}}; ncclIbMergedDevs[5].vProps={2,{2,3}};
 union ibv_gid advertised[2]={}; assert(ncclIbListenerSubnets(advertised,2)==0);
 assert(gidSameSubnet(&advertised[0],&mockGids[0],24));
 assert(gidSameSubnet(&advertised[1],&mockGids[2],24));
 for (int def=0;def<6;def++) {
  int selected=-1; union ibv_gid peer[2]={gid("10.100.224.2"),gid("10.100.225.1")};
  assert(ncclIbFindDevBySubnet(peer,2,def,&selected)==0 && selected==4);
  peer[0]=gid("10.100.227.1"); peer[1]=gid("10.100.226.2");
  assert(ncclIbFindDevBySubnet(peer,2,def,&selected)==0 && selected==5);
 }
 int selected=-1; union ibv_gid peer=gid("10.100.224.2");
 failed=1; assert(ncclIbFindDevBySubnet(&peer,1,4,&selected)==0 && selected==0);
 advertised[0]={}; advertised[1]={}; assert(ncclIbListenerSubnets(advertised,2)!=0); failed=-1;
 ncclIbMergedDevs[3].vProps={2,{0,1}};
 assert(ncclIbFindDevBySubnet(&peer,1,4,&selected)==0 && selected==4);
 ncclIbMergedDevs[3].vProps={1,{3}};
 union ibv_gid missing=gid("10.100.250.1");
 assert(ncclIbFindDevBySubnet(&missing,1,5,&selected)==0 && selected==5);
 mockGids[3]=gid("10.100.226.4"); advertised[0]={}; advertised[1]={};
 assert(ncclIbListenerSubnets(advertised,2)==ncclInvalidUsage);
 prefix=0; assert(ncclIbListenerSubnets(advertised,2)==ncclInvalidArgument);
 assert(ncclIbFindDevBySubnet(&peer,1,0,&selected)==ncclInvalidArgument);
 puts("PASS: cable discovery, reachable pair selection, ties, failed query, missing subnet, overflow, invalid prefix");
}
'''
with tempfile.TemporaryDirectory(prefix='subnet-selection-') as tmp:
    cpp = Path(tmp) / 'test.cc'
    binary = Path(tmp) / 'test'
    cpp.write_text(harness + helpers + cases)
    subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', str(cpp), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
