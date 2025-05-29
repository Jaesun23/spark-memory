# 🚀 Memory One Spark - Quick Start Guide

## 전체 설치 (5분 소요)

### 1. 사전 요구사항
- macOS (Homebrew 설치됨)
- Python 3.11+
- uv (Python 패키지 관리자)

### 2. 원클릭 설치
```bash
git clone https://github.com/yourusername/memory-one-spark.git
cd memory-one-spark
./install.sh
```

이 스크립트가 자동으로:
- ✅ Redis Stack 설치 (이미 있으면 건너뜀)
- ✅ Redis를 커스텀 디렉토리로 설정
- ✅ Python 의존성 설치
- ✅ 설치 검증

### 3. Claude Desktop 연동

`~/Library/Application Support/Claude/claude_desktop_config.json`에 추가:

```json
{
  "mcpServers": {
    "memory-one-spark": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/memory-one-spark",
        "python",
        "-m",
        "src"
      ],
      "env": {
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### 4. 사용 시작

Claude Desktop을 재시작하면 다음 도구들을 사용할 수 있습니다:

- `m_memory`: 메모리 저장/조회/검색
- `m_state`: 상태 관리 및 체크포인트
- `m_admin`: 시스템 관리
- `m_assistant`: AI 어시스턴트 기능

## 문제 해결

### Redis가 시작되지 않는 경우
```bash
# 로그 확인
tail -f ~/dotfiles/config/mcp/memory/redis-stack.log

# 수동으로 시작
launchctl load ~/Library/LaunchAgents/homebrew.mxcl.redis-stack.plist
```

### 모듈이 로드되지 않는 경우
```bash
# 모듈 확인
redis-cli MODULE LIST

# Redis Stack 버전 확인
ls /opt/homebrew/Caskroom/redis-stack-server/
```

### MCP 서버가 연결되지 않는 경우
```bash
# 직접 테스트
python -m src

# 로그 확인
cat ~/Library/Logs/Claude/mcp-server-memory-one-spark.log
```

## 고급 설정

### 커스텀 데이터 디렉토리 사용
`setup/redis-setup.sh`의 10번째 줄을 수정:
```bash
REDIS_DATA_DIR="$HOME/your/custom/path"
```

### 메모리 제한 설정
Redis 설정 파일에 추가:
```conf
maxmemory 2gb
maxmemory-policy allkeys-lru
```

### 보안 설정 활성화
`.env` 파일 생성:
```bash
ENABLE_SECURITY=true
ENCRYPTION_KEY=$(openssl rand -base64 32)
REDIS_PASSWORD=your-secure-password
```

## 다음 단계

1. 📖 [전체 문서](README.md) 읽기
2. 🧪 [예제 코드](docs/examples/) 실행해보기
3. 🛠️ [설정 가이드](docs/REDIS_STACK_CONFIGURATION.md) 참고
4. 💬 문제 발생 시 [이슈](https://github.com/yourusername/memory-one-spark/issues) 등록