# 인증 개요
- 게이트웨이는 하나의 헤더(`X-AnyLLM-Key: Bearer ...`)로 세 가지 토큰을 구분 처리한다.
  - **마스터 키**: 관리자/운영 전용. 키/사용자/예산 등 관리 라우터 보호.
  - **API 키**: 외부 API 호출용. 호출 주체(user_id) 귀속, 예산/사용량 추적의 기준.
  - **Access Token(JWT)**: 웹/관리 세션용 짧은 수명 토큰. 내부적으로 특정 `api_key_id`·`user_id`에 매핑되어 API 키 없이도 호출 가능하게 한다.
- 소셜 로그인 시 신규 가입이면 budgets → users → api_keys → caret_users → access/refresh 토큰을 순서대로 생성한다. 기존 회원이면 기존 budget/user/api_key를 재사용하고 access/refresh 토큰만 갱신한다.

# 토큰·키 요약
- **마스터 키**: 무기한, 관리 엔드포인트 전용. 노출 주의.
- **API 키**: 무기한(또는 만료 설정 가능), 평문은 발급 시 1회만 노출. DB에는 해시만 저장하며 복호화 불가. 분실 시 재발급 필요.
- **Access Token (JWT)**: 짧은 TTL(예: 15~60분). 페이로드에 `sub=user_id`, `api_key_id`, `jti`, `exp` 포함. 서명 검증 + 세션 토큰 상태 확인 필요.
- **Refresh Token**: 긴 TTL(예: 7~30일). 해시로 저장, 회전(Rotate on use) 권장. 만료/폐기 시 재로그인 필요.

# 요청 헤더
- 공통: `X-AnyLLM-Key: Bearer <token>`
  - 값이 마스터 키면 관리자 권한.
  - 값이 API 키면 사용자 귀속 호출.
  - 값이 JWT면 access 토큰으로 간주, 세션 토큰 검증 후 매핑된 `api_key_id`/`user_id`로 처리.

# 플로우
## 가입 + 로그인 (소셜)
1) 소셜 토큰 검증 → 프로필 정규화(`provider`, `role`, `email`, `name`, `avatar`, ...).
2) 요청 본문: `provider`, `access_token`, `email?`, `name?`, `avatar_url?`, `device_type/device_id/os/app_version/user_agent/ip`(선택), `metadata`.
3) `(provider)` 미존재 시 가입:
   - budgets: 기본 예산 레코드 생성(기본값/기간은 설정에 따름).
   - users: 새 user 생성, `budget_id` 연결, `budget_started_at/next_budget_reset_at` 초기화.
   - api_keys: 새 키 생성, `user_id`에 연결(평문 1회 응답).
   - caret_users: 프로바이더 사용자 매핑 생성.
4) 이미 존재 시 기존 budget/user/api_key 재사용.
5) access + refresh 토큰 발급(JWT는 `api_key_id`를 품어 예산/사용량 추적을 이어감). 세션 토큰(`session_tokens`)에는 디바이스/클라이언트 정보(`metadata`)를 저장.
6) 응답 예시 필드: `is_new_user`, `user`, `budget`, `api_key`(신규 시만 평문), `access_token`, `access_token_expires_at`, `refresh_token`, `refresh_token_expires_at`.

## API 호출
- `X-AnyLLM-Key`에 access JWT 또는 API 키를 전달.
- 공통 의존성(예: `verify_jwt_or_api_key_or_master`)에서 토큰 타입 판별 → `api_key_id`·`user_id` 확보 → 예산/사용량 로깅에 사용.
- 마스터 키 사용 시 `user` 필드를 별도로 요구(누적 대상 명시).

## 토큰 갱신 (선제 갱신 권장)
- 클라이언트가 access `exp`를 확인해 만료 몇 분 전 `/v1/auth/refresh` 호출.
- refresh 검증/회전 후 새 access(+새 refresh) 발급, 이전 refresh는 폐기.
- refresh도 만료되면 재로그인 필요.

## 로그아웃
- refresh 토큰을 `revoked_at` 처리(또는 삭제). access는 짧은 TTL이므로 자연 만료, 필요 시 `jti` 블랙리스트로 즉시 차단 가능.
- API 키는 비활성화하지 않음(외부 호출용 지속).

# 라우터 권한 가이드
- 사용자 호출: `v1/chat/completions`만 JWT 또는 API 키 허용(마스터 키도 가능).
- 자기 정보: `/v1/auth/me` → JWT/API 키 허용, 마스터 키는 거부.
- 관리자/운영: `keys`, `users`, `budgets` 관리 등 나머지 기존 라우터 → 마스터 키만 허용(필요 시 별도 관리자 JWT 추가 가능).
- 프로필/사용량: `/v1/profile` → JWT/API 키/마스터 키 모두 허용. 마스터 키는 `user` 쿼리 파라미터로 대상 지정 필수.
  - 기간 집계: `/v1/profile/usage`(group_by=day/week/total, from/to 지정 가능, 요청/토큰/비용 합계)
  - 키 메타: `/v1/profile/keys`(평문 키 미노출, 대상 사용자 키 목록)
  - 로그 조회: `/v1/profile/logs`(필터/페이지네이션: from/to, status, model, provider, endpoint, 토큰/코스트 범위, limit/offset)

# 테이블/모델 요약
- `api_keys`: `id`, `key_hash`, `key_name`, `user_id`, `expires_at`, `is_active`, `metadata`, `created_at`, `last_used_at`.
- `users`: `user_id`, `budget_id`, `spend`, `budget_started_at`, `next_budget_reset_at`, `blocked`, `metadata`, 타임스탬프.
- `budgets`: `budget_id`, `max_budget`, `budget_duration_sec`, 타임스탬프.
- `caret_users`: `id`, `user_id` FK, `provider`, `role`, `email`, `name`, `avatar_url`, `refresh_token?`, `access_token_expires_at`, `last_login_at`, `metadata`, 타임스탬프.
- `session_tokens`: `id/jti`, `user_id`, `api_key_id`, `refresh_token_hash`, `refresh_expires_at`, `revoked_at`, `created_at`, `last_used_at`, `metadata`(디바이스/클라이언트 정보).
- 프로필 API 사용량 집계: 기간별(`24h/7d/30d`) `requests`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `cost` 합계 + 최근 로그 일부.

# 보안/운영 주의
- API 키 평문은 재노출 불가 → 분실 시 재발급/회전으로 대응.
- Access JWT는 짧게, Refresh는 회전 + 재사용 감지로 탈취 대응.
- 마스터 키는 별도 보관·감사, 일반 엔드포인트에 사용 금지.
- 토큰 유출 대비: 만료, 블랙리스트(jti), refresh 회전, 키 비활성화(필요 시) 정책을 문서화.***
