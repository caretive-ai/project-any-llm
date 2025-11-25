# any-llm-gateway 사용 빠른 가이드

## 1) 설치
- 가상환경 권장:
  ```bash
  uv venv .venv
  source .venv/bin/activate
  uv pip install '.[gateway]'         # 로컬 소스에서
  # 또는 PyPI: uv pip install 'any-llm-sdk[gateway]'
  ```
- 프로바이더별 extras 필요 시 추가:
  ```bash
  uv pip install '.[gemini]'   # google-genai 설치
  uv pip install '.[openai]'   # 예시
  ```
  zsh는 따옴표 또는 `\[`로 글로빙 회피.

## 2) 설정(config.yml)
- docker/config.example.yml 참고. 최소 필드:
  ```yaml
  database_url: "postgresql://.../gateway"
  host: "0.0.0.0"
  port: 8000
  master_key: YOUR_MASTER_KEY
  providers:
    gemini:
      api_key: YOUR_GEMINI_API_KEY   # or GOOGLE_API_KEY
    openai:
      api_key: YOUR_OPENAI_API_KEY
  pricing: {}  # 필요 없으면 빈 dict로 두거나 제거
  ```
- pricing이 None이면 Pydantic 에러 → 빈 dict로 설정.

## 3) 실행
- 로컬: `any-llm-gateway serve --config config.yml`
  - env override: `GATEWAY_HOST`, `GATEWAY_PORT`, `GATEWAY_DATABASE_URL`.
- Docker: `docker compose -f docker/docker-compose.yml up --build`
  - 이미지 단독: `docker run -p 8000:8000 <image> any-llm-gateway serve`

## 4) 키 발급 (마스터 키 사용)
```bash
curl -X POST http://localhost:8000/v1/keys \
  -H "Content-Type: application/json" \
  -H "X-AnyLLM-Key: Bearer <MASTER_KEY>" \
  -d '{"key_name":"demo"}'
```
- 응답의 `key` 값이 평문 API 키(`gw-...`). DB에는 해시만 저장되므로 이 값을 보관.
- `GET /v1/keys`에서는 평문 키를 다시 볼 수 없다.

## 5) 채팅 호출
- 비스트리밍:
  ```bash
  curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "X-AnyLLM-Key: Bearer <API_KEY_OR_MASTER>" \
    -d '{
      "model": "gemini:gemini-3-pro-preview",
      "messages": [{"role":"user","content":"Hello"}]
    }'
  ```
- 스트리밍(SSE):
  ```bash
  curl -N -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "X-AnyLLM-Key: Bearer <API_KEY_OR_MASTER>" \
    -d '{
      "model": "gemini:gemini-3-pro-preview",
      "stream": true,
      "messages": [{"role":"user","content":"Hello"}]
    }'
  ```
- 모델은 `provider:model` 형식 권장. `config.yml`의 `providers`에 있는 것만 호출 가능.

## 6) 마스터 키 vs API 키
- 마스터 키로 채팅 호출 시 `user` 필드가 필수:
  ```json
  {"user": "demo-user", ...}
  ```
- API 키(발급된 `gw-...`)로 호출하면 `user`를 생략해도 키에 연결된 사용자를 사용.

## 7) 자주 보는 오류
- `When using master key, 'user' field is required`: 마스터 키 호출에 `user` 누락 → `user` 추가하거나 API 키 사용.
- `No module named 'google'` 등: 해당 프로바이더 extras 미설치 → `pip install '.[gemini]'` 등으로 설치 후 서버 재시작.
- Pydantic `pricing` dict 에러: `pricing: {}`로 설정하거나 필드 제거.

## 8) 헬스 체크
```bash
curl http://localhost:8000/health
```

## 9) 기타 라우트
- `/v1/keys`, `/v1/users`, `/v1/budgets`, `/v1/pricing`, `/health` 등은 `X-AnyLLM-Key: Bearer <MASTER_KEY>`로 접근.

## 10) 파일 수정후 결과 확인
uv pip install '.[gateway]'
any-llm-gateway serve --config config.yml