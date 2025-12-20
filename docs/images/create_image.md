## 이미지 생성 흐름 정리

이 문서는 최근 추가된 이미지 생성 관련 변경사항을 정리합니다.

1. **마크다운 렌더러에서 생성 이미지 클릭**  
   - `webview-ui/src/components/common/MarkdownBlock.tsx`에서 data URL 이미지에 `img` 컴포넌트를 커스터마이징하여 `FileServiceClient.openImage(StringRequest.create({ value: src }))`를 실행하도록 함.  
   - 툴팁/aria-label 문자열은 각 로케일(`en/ko/ja/zh`)의 `chat.json` `markdownBlock.openImageInEditor` 항목을 참고하여 다국어 지원.  
   - 이 클릭은 에디터 새로운 탭에서 이미지를 여는 흐름이며 기존 `caret` 로고와 겹치지 않도록 적절하게 cursor/role 도셋팅.

2. **Gemini 이미지 게이트웨이에서의 사용량/비용 처리**  
   - `routes/image.py`는 단일 이미지 생성 요청과 스트리밍 요청 모두 `UsageLog` 및 `User.spend`를 갱신하도록 `charge_usage_cost`, `_log_image_usage`, `_set_usage_cost`, `_add_user_spend`를 적용.  
   - `UsageAccumulator`를 도입하여 스트리밍 chunk마다 `prompt/candidate/total/thought` 토큰을 누적하고, 최종적으로 누적값만 과금에 반영하여 중복 청구를 방지.  
   - `thought_tokens`를 `completion_tokens`에 포함시키며, total 계산도 누적된 생각 토큰과 prompt+completion 합을 기준으로 함.  
   - 스트리밍 및 비스트리밍 요청 모두 `UsageLog.cost`를 업데이트하고, user의 `spend`도 함께 증가시켜 실제 비용 반영.
   - thought_tokens 비용은 cached_price_per_million으로 처리.

3. **로그/디버깅 정보**  
   - 스트리밍 chunks의 token usage, mime type, data 길이 등 상세 로그를 추가하여 현상 추적 가능.  
   - `usage_metadata`가 없을 경우 `usage` 객체를 보조하고, `SimpleNamespace` 형태로 통일하여 cost 계산에 이용.

4. **기타**  
   - `package.json` 등 스크립트 변경 없이 문서화만 진행했습니다.  
   - 테스트는 별도로 실행하지 않았습니다 (요청 없음).
