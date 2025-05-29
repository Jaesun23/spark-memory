# MCP 2025 Quick Reference Guide

## 🚨 코드 작성 시 반드시 체크!

### 1. JSON-RPC로 변경
```python
# ❌ 기존 방식
@app.tool()
async def m_memory(...):
    return result

# ✅ 새로운 방식 (JSON-RPC 래퍼 필요)
@app.tool()
async def m_memory(...):
    # JSON-RPC request 처리
    # id 추적
    # 표준 response 형식
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result
    }
```

### 2. 프리미티브 추가
```python
# Tools (기존)
@app.tool()

# Prompts (신규)
@app.prompt()

# Resources (신규)
@app.resource()
```

### 3. 보안 체크
```python
# 도구 실행 전
if tool.dangerous:
    consent = await get_user_consent()
    if not consent:
        raise SecurityError()
```

### 4. 양방향 통신
- 서버에서 클라이언트로 이벤트 보내기
- 스트리밍 업데이트
- 실시간 진행상황

## 📝 작업 순서
1. JSON-RPC 핸들러 먼저 구현
2. 기존 Tools를 새 형식으로 마이그레이션
3. Prompts, Resources 추가
4. 보안 레이어 구축
5. 테스트, 테스트, 테스트!

---
더 자세한 내용은:
- /docs/mcp-latest-changes-2025.md
- /docs/phase3-checklist-mcp2025.md
