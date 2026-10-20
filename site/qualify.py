import base64
import concurrent.futures
import json
import os
import gzip
import hashlib
import secrets
import threading
from pathlib import Path
import re
import struct
import sys
import time
import urllib.request
import zlib

BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8000'
SELECTED = set(sys.argv[2:] or ('arithmetic', 'tools', 'vision', 'reasoning'))
BACKEND_BASE = os.environ.get('VLLM_QUALIFY_BACKEND_BASE', 'http://127.0.0.1:8000')
MODEL = os.environ.get('VLLM_QUALIFY_MODEL', 'deepseek-v4.1-flash')
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('VLLM_QUALIFY_OUT', ROOT / 'results/private/site-qualification/direct'))
OUT.mkdir(parents=True, exist_ok=True)


def request(path, payload, base=BASE):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=2700) as response:
        return json.load(response)


def chat(messages, **kwargs):
    return request('/v1/chat/completions', {
        'model': MODEL, 'messages': messages, 'temperature': 0, 'max_tokens': 256,
        'chat_template_kwargs': {'thinking': False}, **kwargs,
    })


def content(response):
    return response['choices'][0]['message'].get('content') or ''


def save(name, response):
    (OUT / (name + '.json')).write_text(json.dumps(response, indent=2))


def run(name, operation):
    started = time.monotonic()
    try:
        result = operation()
        save(name, result)
        print(json.dumps({'test': name, 'passed': True, 'seconds': round(time.monotonic() - started, 2)}), flush=True)
    except Exception as error:
        save(name + '-failure', {'error': str(error)})
        raise


def arithmetic():
    result = chat([{'role': 'user', 'content': 'What is 47 times 83? Reply with the integer only.'}])
    assert re.search(r'\b3901\b', content(result)), result
    return result


def tools():
    definitions = [{'type': 'function', 'function': {'name': 'add', 'description': 'Add two integers', 'parameters': {
            'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
            'required': ['a', 'b'], 'additionalProperties': False}}}]
    messages = [{'role': 'user', 'content': 'Use the add tool with a=17 and b=25 to obtain the result. Do not calculate it yourself.'}]
    result = chat(messages, tools=definitions, tool_choice='auto')
    calls = result['choices'][0]['message']['tool_calls']
    assert calls[0]['function']['name'] == 'add'
    assert json.loads(calls[0]['function']['arguments']) == {'a': 17, 'b': 25}
    messages.extend([result['choices'][0]['message'],
                     {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': '{"result":42}'}])
    continued = chat(messages, tools=definitions, tool_choice='auto')
    assert re.search(r'\b42\b', content(continued)), continued
    return {'call': result, 'continuation': continued}


def vision():
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    width = 128
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, width, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress((b'\0' + bytes((255, 0, 0)) * width) * width)) + chunk(b'IEND', b'')
    uri = 'data:image/png;base64,' + base64.b64encode(png).decode()
    result = chat([{'role': 'user', 'content': [
        {'type': 'text', 'text': 'What color fills this image? Reply with one color word.'},
        {'type': 'image_url', 'image_url': {'url': uri}},
    ]}])
    assert re.search(r'\bred\b', content(result), re.I), result
    return result


def reasoning():
    result = chat([{'role': 'user', 'content': 'Solve 23 times 19.'}],
        chat_template_kwargs={'thinking': True}, max_tokens=2048)
    message = result['choices'][0]['message']
    assert re.search(r'\b437\b', content(result)), result
    assert message.get('reasoning_content') or message.get('reasoning'), result
    return result


def history():
    key = 'violet-7391-copper-2864'
    filler = '\n'.join(f'Record {i}: the cedar archive holds item {i * 17 + 31} for routine inspection.' for i in range(700))
    messages = [{'role': 'user', 'content': f'The secret key is {key}.\n{filler}\nReturn the secret key only.'}]
    first = chat(messages)
    assert key in content(first), first
    messages.extend([{'role': 'assistant', 'content': content(first)}, {'role': 'user', 'content': 'Repeat the same key only.'}])
    second = chat(messages)
    assert key in content(second), second
    details = second.get('usage', {}).get('prompt_tokens_details') or {}
    assert details.get('cached_tokens', 0) > 0, second
    return {'initial': first, 'extension': second}


def parallel():
    cases = [(31, 17), (43, 19), (53, 29), (61, 37)]
    def one(pair):
        a, b = pair
        result = chat([{'role': 'user', 'content': f'Compute {a} + {b}. Start with the answer, then give a five-sentence explanation of decimal addition and carrying.'}])
        assert re.search(rf'\b{a + b}\b', content(result)), result
        assert result['usage']['completion_tokens'] >= 32, result
        assert not re.search(r'[\u3000-\u9fff\uac00-\ud7af]', content(result)), result
        return result
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        return list(pool.map(one, cases))


def streaming_sixteen():
    barrier = threading.Barrier(16)
    done = threading.Event()
    samples = []
    def monitor():
        while not done.is_set():
            try:
                with urllib.request.urlopen(BACKEND_BASE + '/metrics', timeout=5) as response:
                    lines = response.read().decode().splitlines()
                samples.append(sum(float(line.split()[-1]) for line in lines
                                   if line.startswith('vllm:num_requests_running')))
            except Exception as error:
                samples.append({'error': str(error)})
            done.wait(0.05)
    def one(i):
        first = 10000 + i * 1000
        expected = list(range(first, first + 80))
        payload = {'model': MODEL, 'messages': [{'role': 'user', 'content':
            f'Print every integer from {first} through {first+79} inclusive, in ascending order, separated by spaces. Output only the integers.'}],
            'temperature': 0, 'max_tokens': 768, 'stream': True,
            'stream_options': {'include_usage': True}, 'chat_template_kwargs': {'thinking': False}}
        req = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        barrier.wait(timeout=30)
        text, finished, ended, usage = '', None, False, None
        with urllib.request.urlopen(req, timeout=900) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                data = line[6:].strip()
                if data == b'[DONE]':
                    ended = True
                    break
                chunk = json.loads(data)
                assert 'error' not in chunk, chunk
                usage = chunk.get('usage') or usage
                for choice in chunk.get('choices', []):
                    text += choice.get('delta', {}).get('content') or ''
                    finished = choice.get('finish_reason') or finished
        result = {'id': i, 'text': text, 'finish_reason': finished, 'done': ended, 'usage': usage}
        save(f'stream16-{i:02d}', result)
        assert ended and finished == 'stop', result
        assert [int(n) for n in re.findall(r'\d+', text)] == expected, result
        return result
    sampler = threading.Thread(target=monitor)
    sampler.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(16) as pool:
            results = list(pool.map(one, range(16)))
    finally:
        done.set()
        sampler.join()
        save('stream16-scheduler', samples)
    peak = max((n for n in samples if isinstance(n, float)), default=0)
    assert peak >= 16, {'scheduler_peak': peak, 'samples': samples}
    return {'requests': results, 'scheduler_peak': peak}


def near_million():
    roots = list((ROOT / '.work/build').glob('*/src/vllm/vllm'))
    assert len(roots) == 1, roots
    pieces, length = [], 0
    for path in sorted(roots[0].rglob('*.py')):
        text = f'\n# file: {path.relative_to(roots[0])}\n{path.read_text(errors="replace")}\n'
        pieces.append(text)
        length += len(text)
        if length >= 8000000:
            break
    corpus = ''.join(pieces)
    assert len(corpus) >= 6000000, len(corpus)
    markers = {letter: f'{letter}-copper-{secrets.token_hex(6)}' for letter in 'ABC'}
    salt = secrets.token_hex(16)
    def messages_for(size):
        text = corpus[:size]
        for letter, depth in reversed(list(zip('ABC', (0.1, 0.5, 0.9)))):
            at = text.rfind('\n', 0, int(size * depth)) + 1
            text = text[:at] + f'\n# ARCHIVE_{letter}={markers[letter]}\n' + text[at:]
        return [{'role': 'user', 'content': f'Archive audit {salt}.\n' + text +
                 '\nReturn a JSON object with keys A, B and C containing the exact ARCHIVE_A, ARCHIVE_B and ARCHIVE_C values above. Output only the JSON object.'}]
    target = 1028000
    lo, hi = 0, len(corpus)
    messages = None
    for _ in range(24):
        size = (lo + hi) // 2
        candidate = messages_for(size)
        count = request('/tokenize', {'model': 'deepseek-v4.1-flash', 'messages': candidate,
                        'chat_template_kwargs': {'thinking': False}}, base=BACKEND_BASE)['count']
        if target-1024 <= count <= target:
            messages = candidate
            break
        if count > target:
            hi = size-1
        else:
            lo = size+1
    assert messages is not None, 'Could not fit near1M request'
    payload = {'model': MODEL, 'messages': messages, 'temperature': 0, 'max_tokens': 128,
               'chat_template_kwargs': {'thinking': False}, 'cache_salt': salt}
    raw = json.dumps(payload).encode()
    with gzip.open(OUT / 'near1m-request.json.gz', 'wb') as handle:
        handle.write(raw)
    save('near1m-input', {'sha256': hashlib.sha256(raw).hexdigest(), 'tokenized_prompt': count,
                         'markers': markers, 'cache_salt': salt})
    print(json.dumps({'event': 'near1m-start', 'tokenized_prompt': count}), flush=True)
    reply = request('/v1/chat/completions', payload)
    save('near1m-response', reply)
    assert reply['choices'][0]['finish_reason'] == 'stop', reply
    assert 1000000 <= reply['usage']['prompt_tokens'] <= 1048448, reply['usage']
    assert all(value in content(reply) for value in markers.values()), reply
    return reply


tests = [('arithmetic', arithmetic), ('tools', tools), ('vision', vision),
         ('reasoning', reasoning), ('history-prefix', history), ('parallel-four', parallel),
         ('stream16', streaming_sixteen), ('near1m', near_million)]
assert SELECTED <= {name for name, _ in tests}, SELECTED
for name, operation in tests:
    if name in SELECTED:
        run(name, operation)
