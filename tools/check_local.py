"""실행 중인 로컬 RAG의 준비 상태와 실제 답변을 확인한다.

python tools/check_local.py --question "SR650 V4 최대 메모리 용량은?"
"""
import argparse
import json

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--question', default='SR630 V4 최대 메모리 용량은?')
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=300) as client:
        ready = client.get('/ready')
        ready.raise_for_status()
        response = client.post('/ask', json={'question': args.question, 'top_k': 5})
        response.raise_for_status()
        result = response.json()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result['sources']:
            raise SystemExit('답변 근거를 찾지 못했습니다. 문서 색인과 검색 결과를 확인하세요.')
        unknown = client.post('/ask', json={'question': 'SR999 V9 스펙은?'})
        unknown.raise_for_status()
        assert unknown.json()['answer'] == '제공된 문서에서 찾지 못했습니다.'
        assert not unknown.json()['sources']
        print('준비 상태·실제 답변·미등록 모델 응답 확인 완료')


if __name__ == '__main__':
    main()
