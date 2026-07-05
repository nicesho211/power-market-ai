#!/usr/bin/env python
"""패키지 설치 상태 확인"""

import sys
import importlib

packages = [
    'langchain',
    'langchain_openai',
    'langchain_community',
    'langgraph',
    'chromadb',
    'streamlit',
    'plotly',
    'pandas',
    'pydantic',
    'pydantic_settings',
    'requests',
    'python_dotenv',
    'pymupdf4llm'
]

print("=" * 60)
print("패키지 설치 상태 확인")
print("=" * 60)

installed = []
missing = []

for pkg in packages:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, '__version__', 'unknown')
        installed.append((pkg, version))
        print(f"✅ {pkg:<25} - {version}")
    except ImportError:
        missing.append(pkg)
        print(f"❌ {pkg:<25} - 미설치")

print("=" * 60)
print(f"설치됨: {len(installed)}/{len(packages)}")
if missing:
    print(f"미설치: {', '.join(missing)}")
    print("\n설치 명령:")
    print(f"  pip install {' '.join(missing)}")
else:
    print("✅ 모든 패키지 설치됨!")
print("=" * 60)
