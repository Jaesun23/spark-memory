# 개발자 가이드

## 프로젝트 설정

### 타입 체킹 설정

이 프로젝트는 코드 품질을 위해 엄격한 타입 체킹을 사용합니다.

#### mypy 설정 (`pyproject.toml`)

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
exclude = [
    "tests/",    # 테스트 파일은 타입 힌트 선택사항
    "test_*.py",
]
```

#### 중요 사항

1. **소스 코드 (`src/` 디렉토리)**
   - ✅ 모든 함수, 메서드, 클래스에 타입 힌트 필수
   - ✅ mypy의 엄격한 검사 적용
   - ✅ pre-commit hook으로 자동 검사

2. **테스트 코드 (`tests/` 디렉토리)**
   - ⚠️ 타입 힌트 선택사항
   - ⚠️ mypy 검사에서 제외됨
   - ⚠️ 하지만 가능하면 타입 힌트 추가 권장

### Pre-commit 설정

`.pre-commit-config.yaml`에서 mypy hook 설정:

```yaml
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.15.0
  hooks:
    - id: mypy
      additional_dependencies: [types-redis]
      exclude: ^tests/  # 테스트 파일 제외
```

### 개발 워크플로우

1. **코드 작성**
   ```python
   # src/memory/engine.py - 타입 힌트 필수!
   from typing import Optional, List, Dict, Any

   async def save_memory(
       paths: List[str],
       content: Any,
       options: Optional[Dict[str, Any]] = None
   ) -> str:
       """메모리 저장."""
       # 구현...
   ```

2. **테스트 작성**
   ```python
   # tests/unit/test_engine.py - 타입 힌트 선택사항
   import pytest

   # 타입 힌트 없어도 OK
   def test_save_memory():
       result = save_memory(["test"], "content")
       assert result == "expected"

   # 하지만 추가하면 더 좋음
   def test_save_memory_typed() -> None:
       result: str = save_memory(["test"], "content")
       assert result == "expected"
   ```

3. **커밋 전 검사**
   ```bash
   # 자동 실행됨 (pre-commit hook)
   git commit -m "✨ Add new feature"

   # 수동 실행
   pre-commit run --all-files

   # 특정 파일만 검사
   mypy src/memory/engine.py
   ```

### 문제 해결

#### "Function is missing a return type annotation" 에러
- **소스 코드**: 반드시 수정 필요
- **테스트 코드**: 무시됨 (mypy exclude 설정)

#### pre-commit 실패 시
```bash
# hook 일시적으로 스킵 (권장하지 않음)
git commit --no-verify

# 특정 hook만 스킵
SKIP=mypy git commit
```

### 코드 리뷰 체크리스트

- [ ] 소스 코드에 타입 힌트가 있는가?
- [ ] 테스트가 작성되었는가?
- [ ] pre-commit이 통과했는가?
- [ ] 문서가 업데이트되었는가?

## 추가 리소스

- [mypy 문서](https://mypy.readthedocs.io/)
- [Python 타입 힌트 가이드](https://docs.python.org/3/library/typing.html)
- [pre-commit 문서](https://pre-commit.com/)
