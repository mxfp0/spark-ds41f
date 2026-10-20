FROM deepseek1017-vllm:r6x-tp4-64k AS tilelang
FROM vllm/vllm-openai:v0.32.0@sha256:6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc liburing-dev patch patchutils pkg-config python3-dev && rm -rf /var/lib/apt/lists/*

RUN uv pip uninstall --system flashinfer-jit-cache \
        flashinfer-jit-cache-sm80 flashinfer-jit-cache-sm89 flashinfer-jit-cache-sm90a \
        flashinfer-jit-cache-sm100a flashinfer-jit-cache-sm103a flashinfer-jit-cache-sm110a \
    && uv pip install --system --no-deps b12x==1.5.1 flashinfer-python==0.7.1 \
    && uv pip install --system --no-deps flashinfer-cubin==0.7.1 --index-url https://flashinfer.ai/whl \
    && uv pip install --system --no-deps --extra-index-url https://flashinfer.ai/whl/cu130/ \
        flashinfer-jit-cache==0.7.1+cu130 flashinfer-jit-cache-sm120f==0.7.1+cu130 \
        flashinfer-jit-cache-sm121a==0.7.1+cu130

RUN uv pip uninstall --system tilelang
COPY --from=tilelang /usr/local/lib/python3.12/dist-packages/tilelang /usr/local/lib/python3.12/dist-packages/tilelang
COPY --from=tilelang /usr/local/lib/python3.12/dist-packages/tilelang-0.1.15.dist-info /usr/local/lib/python3.12/dist-packages/tilelang-0.1.15.dist-info
RUN echo '04c6efebdd7a578c9e579977b1393a3623a19fbfbd780e2fb4a89ce86bb7b1cd  /usr/local/lib/python3.12/dist-packages/tilelang/lib/libtilelang.so' | sha256sum --check --status

COPY --from=tilelang /usr/local/lib/python3.12/dist-packages/tile_kernels /usr/local/lib/python3.12/dist-packages/tile_kernels
COPY --from=tilelang /usr/local/lib/python3.12/dist-packages/tile_kernels-2.0.0.dist-info /usr/local/lib/python3.12/dist-packages/tile_kernels-2.0.0.dist-info
COPY --from=tilelang /usr/local/lib/python3.12/dist-packages/sparknet /usr/local/lib/python3.12/dist-packages/sparknet
COPY --from=tilelang /usr/local/lib/python3.12/dist-packages/dgx_spark_networking-0.3.0.dist-info /usr/local/lib/python3.12/dist-packages/dgx_spark_networking-0.3.0.dist-info
COPY site/vllm032 /opt/spark-site/vllm032
COPY experiments/2026-10-18-local-engram/local_engram.py \
    experiments/2026-10-18-local-engram/local-plan.json /opt/spark-site/local-engram/
ENV PYTHONPATH=/opt/spark-site:/opt/spark-site/local-engram
ARG VLLM_SITE_TREE
LABEL local.spark3.vllm.tree="${VLLM_SITE_TREE}"
LABEL local.spark3.tilelang.tree="222b895d04ee6cd75179155d06e0c0d450636f51"
LABEL local.spark3.tile_kernels.tree="cb2fa41c8af290204c81c547ed678ebf957cca53"
LABEL local.spark3.sparknet.tree="f72599fa212e894d67ee114553e628ab05daf2da"

RUN python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent.parent)' > /tmp/vllm-root \
    && while IFS= read -r name; do \
        filterdiff --include='a/vllm/*' --include='b/vllm/*' \
            "/opt/spark-site/vllm032/$name" > /tmp/vllm-site.patch; \
        patch --batch --forward -d "$(cat /tmp/vllm-root)" -p1 < /tmp/vllm-site.patch || exit 1; \
    done < /opt/spark-site/vllm032/series \
    && VLLM_PLUGINS='' python3 -c 'from vllm.models.deepseek_v4_1.nvidia.model import DeepseekV41LLMForCausalLM; from vllm.models.deepseek_v4_1.engram_rows._native import load; load()'
