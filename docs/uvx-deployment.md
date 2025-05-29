# uvx 배포 가이드

## 개요

Memory One Spark는 uvx를 통해 간편하게 배포하고 실행할 수 있습니다. Docker 없이도 Redis Stack만 로컬에 설치하면 바로 사용 가능합니다.

## 배포 방식

### 1. PyPI 배포 (권장)

```bash
# PyPI에 패키지 업로드
uv build
uv publish

# 사용자가 실행
uvx memory-one-spark
```

### 2. GitHub 직접 실행

```bash
# main 브랜치에서 직접 실행
uvx --from git+https://github.com/yourusername/memory-one-spark.git memory-one-spark

# 특정 태그/브랜치 실행
uvx --from git+https://github.com/yourusername/memory-one-spark.git@v0.1.0 memory-one-spark
```

## Redis Stack 설치 가이드

### macOS (Homebrew)

```bash
# Redis Stack tap 추가
brew tap redis-stack/redis-stack

# 설치
brew install redis-stack

# 서비스로 실행 (자동 시작)
brew services start redis-stack

# 또는 수동 실행
redis-stack-server
```

### Ubuntu/Debian

```bash
# GPG 키 추가
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg

# 저장소 추가
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list

# 설치
sudo apt-get update
sudo apt-get install redis-stack-server

# 서비스 시작
sudo systemctl start redis-stack-server
sudo systemctl enable redis-stack-server
```

### Windows

1. [Redis Stack 다운로드 페이지](https://redis.io/download/#redis-stack-downloads) 방문
2. Windows용 설치 파일 다운로드
3. 설치 후 서비스로 실행

### Docker (개발 참고용)

```bash
# 개발 환경에서 빠른 테스트용
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

## Claude Desktop 설정

### 기본 설정

`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) 또는 해당 OS의 설정 파일에 추가:

```json
{
  "mcpServers": {
    "memory-one-spark": {
      "command": "uvx",
      "args": ["memory-one-spark"],
      "env": {
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### 고급 설정 (보안 활성화)

```json
{
  "mcpServers": {
    "memory-one-spark": {
      "command": "uvx",
      "args": ["memory-one-spark"],
      "env": {
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "REDIS_PASSWORD": "your-redis-password",
        "ENABLE_SECURITY": "true",
        "ENCRYPTION_KEY": "your-32-byte-encryption-key-here",
        "LOG_LEVEL": "INFO",
        "LOG_FORMAT": "json"
      }
    }
  }
}
```

## 문제 해결

### Redis Stack 연결 오류

```bash
# Redis Stack 상태 확인
redis-cli ping
# 응답: PONG

# 모듈 확인
redis-cli MODULE LIST
# ReJSON, RediSearch, RedisTimeSeries가 표시되어야 함
```

### 권한 오류

```bash
# uvx 캐시 정리
uvx cache clean

# 재설치
uvx --reinstall memory-one-spark
```

### 로그 확인

```bash
# 환경 변수로 로그 레벨 조정
LOG_LEVEL=DEBUG uvx memory-one-spark
```

## 버전 관리

### 특정 버전 실행

```bash
# 특정 버전
uvx memory-one-spark==0.1.0

# 최소 버전
uvx "memory-one-spark>=0.1.0"

# 프리릴리즈 포함
uvx --pre memory-one-spark
```

### 업데이트

```bash
# 최신 버전으로 업데이트
uvx --upgrade memory-one-spark
```

## 개발자 노트

### 로컬 개발 버전 테스트

```bash
# 프로젝트 루트에서
uv build
uvx --from dist/*.whl memory-one-spark
```

### CI/CD 통합

GitHub Actions 예시:

```yaml
name: Publish to PyPI

on:
  release:
    types: [created]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Install uv
      uses: astral-sh/setup-uv@v3
    - name: Build and publish
      env:
        UV_PUBLISH_TOKEN: ${{ secrets.PYPI_API_TOKEN }}
      run: |
        uv build
        uv publish
```