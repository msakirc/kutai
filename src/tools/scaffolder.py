"""Project scaffolding — generates boilerplate code for common stacks.

Instead of having a small LLM generate boilerplate (which it does poorly),
pre-generate standard project structures via templates.
"""

import os
from pathlib import Path
from src.infra.logging_config import get_logger

logger = get_logger("tools.scaffolder")

STACKS = {
    "fastapi": {
        "description": "FastAPI + SQLAlchemy + Alembic + Docker",
        "files": {
            "app/main.py": 'from fastapi import FastAPI\n\napp = FastAPI(title="{project_name}")\n\n@app.get("/health")\ndef health():\n    return {{"status": "ok"}}\n',
            "app/__init__.py": "",
            "app/config.py": 'import os\n\nDATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")\nSECRET_KEY = os.getenv("SECRET_KEY", "changeme")\n',
            "app/models/__init__.py": "",
            "app/routers/__init__.py": "",
            "requirements.txt": "fastapi>=0.100.0\nuvicorn[standard]>=0.20.0\nsqlalchemy>=2.0.0\nalembic>=1.12.0\npython-dotenv>=1.0.0\n",
            "Dockerfile": "FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nCMD [\"uvicorn\", \"app.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n",
            "docker-compose.yml": 'version: "3.8"\nservices:\n  app:\n    build: .\n    ports:\n      - "8000:8000"\n    env_file: .env\n',
            ".env.example": "DATABASE_URL=sqlite:///./app.db\nSECRET_KEY=changeme\n",
            ".gitignore": "__pycache__/\n*.pyc\n.env\n*.db\n.venv/\n",
        },
    },
    "nextjs": {
        "description": "Next.js 14 + TypeScript + Tailwind",
        "files": {
            "package.json": '{{\n  "name": "{project_name}",\n  "scripts": {{\n    "dev": "next dev",\n    "build": "next build",\n    "start": "next start"\n  }},\n  "dependencies": {{\n    "next": "^14.0.0",\n    "react": "^18.0.0",\n    "react-dom": "^18.0.0"\n  }},\n  "devDependencies": {{\n    "typescript": "^5.0.0",\n    "@types/react": "^18.0.0",\n    "tailwindcss": "^3.3.0",\n    "autoprefixer": "^10.0.0",\n    "postcss": "^8.0.0"\n  }}\n}}\n',
            "tsconfig.json": '{{\n  "compilerOptions": {{\n    "target": "es5",\n    "lib": ["dom", "es2017"],\n    "jsx": "preserve",\n    "strict": true,\n    "module": "esnext",\n    "moduleResolution": "bundler",\n    "paths": {{ "@/*": ["./src/*"] }}\n  }}\n}}\n',
            "src/app/layout.tsx": 'export default function RootLayout({{ children }}: {{ children: React.ReactNode }}) {{\n  return (\n    <html lang="en">\n      <body>{{children}}</body>\n    </html>\n  )\n}}\n',
            "src/app/page.tsx": 'export default function Home() {{\n  return <main><h1>{project_name}</h1></main>\n}}\n',
            ".gitignore": "node_modules/\n.next/\n.env\n",
        },
    },
    "expo": {
        "description": "Expo + React Native + TypeScript",
        "files": {
            "package.json": '{{\n  "name": "{project_name}",\n  "main": "expo-router/entry",\n  "scripts": {{\n    "start": "expo start",\n    "android": "expo start --android",\n    "ios": "expo start --ios"\n  }},\n  "dependencies": {{\n    "expo": "~50.0.0",\n    "expo-router": "~3.0.0",\n    "react": "18.2.0",\n    "react-native": "0.73.0"\n  }}\n}}\n',
            "app/_layout.tsx": 'import {{ Stack }} from "expo-router";\nexport default function Layout() {{\n  return <Stack />;\n}}\n',
            "app/index.tsx": 'import {{ Text, View }} from "react-native";\nexport default function Home() {{\n  return <View><Text>{project_name}</Text></View>;\n}}\n',
            "tsconfig.json": '{{\n  "extends": "expo/tsconfig.base",\n  "compilerOptions": {{ "strict": true }}\n}}\n',
            ".gitignore": "node_modules/\n.expo/\n*.jks\n*.p8\n*.p12\n*.key\n*.mobileprovision\n",
        },
    },
    "fastify": {
        "description": "Fastify + TypeScript + Prisma + Docker",
        "files": {
            "package.json": '{{\n  "name": "{project_name}",\n  "scripts": {{\n    "dev": "tsx watch src/server.ts",\n    "build": "tsc",\n    "start": "node dist/server.js"\n  }},\n  "dependencies": {{\n    "fastify": "^4.26.0",\n    "@fastify/cors": "^9.0.0",\n    "@fastify/helmet": "^11.0.0",\n    "@prisma/client": "^5.10.0",\n    "dotenv": "^16.4.0"\n  }},\n  "devDependencies": {{\n    "typescript": "^5.3.0",\n    "tsx": "^4.7.0",\n    "@types/node": "^20.11.0",\n    "prisma": "^5.10.0"\n  }}\n}}\n',
            "tsconfig.json": '{{\n  "compilerOptions": {{\n    "target": "ES2022",\n    "module": "NodeNext",\n    "moduleResolution": "NodeNext",\n    "outDir": "dist",\n    "rootDir": "src",\n    "strict": true,\n    "esModuleInterop": true,\n    "skipLibCheck": true\n  }},\n  "include": ["src/**/*"]\n}}\n',
            "src/server.ts": 'import Fastify from "fastify";\nimport cors from "@fastify/cors";\nimport helmet from "@fastify/helmet";\nimport "dotenv/config";\n\nconst app = Fastify({{ logger: true }});\n\napp.register(cors);\napp.register(helmet);\n\napp.get("/health", async () => ({{ status: "ok" }}));\n\nconst start = async () => {{\n  const port = Number(process.env.PORT) || 3000;\n  await app.listen({{ port, host: "0.0.0.0" }});\n  console.log(`Server running on port ${{port}}`);\n}};\n\nstart();\n',
            "src/plugins/.gitkeep": "",
            "src/routes/.gitkeep": "",
            "prisma/schema.prisma": 'generator client {{\n  provider = "prisma-client-js"\n}}\n\ndatasource db {{\n  provider = "postgresql"\n  url      = env("DATABASE_URL")\n}}\n',
            "Dockerfile": "FROM node:20-slim\nWORKDIR /app\nCOPY package*.json ./\nRUN npm ci --production=false\nCOPY . .\nRUN npx prisma generate\nRUN npm run build\nCMD [\"node\", \"dist/server.js\"]\n",
            ".env.example": 'PORT=3000\nDATABASE_URL="postgresql://user:pass@localhost:5432/mydb"\n',
            ".gitignore": "node_modules/\ndist/\n.env\n",
        },
    },
    "flask": {
        "description": "Flask + SQLAlchemy + Docker",
        "files": {
            "app/__init__.py": 'from flask import Flask\n\ndef create_app():\n    app = Flask(__name__)\n    app.config.from_object("config.Config")\n    return app\n',
            "app/routes.py": 'from flask import Blueprint, jsonify\n\nbp = Blueprint("main", __name__)\n\n@bp.route("/health")\ndef health():\n    return jsonify(status="ok")\n',
            "config.py": 'import os\n\nclass Config:\n    SECRET_KEY = os.getenv("SECRET_KEY", "changeme")\n    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///app.db")\n',
            "requirements.txt": "flask>=3.0.0\nflask-sqlalchemy>=3.1.0\ngunicorn>=21.0.0\npython-dotenv>=1.0.0\n",
            "Dockerfile": "FROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nCMD [\"gunicorn\", \"app:create_app()\", \"-b\", \"0.0.0.0:5000\"]\n",
            ".gitignore": "__pycache__/\n*.pyc\n.env\n*.db\n.venv/\n",
        },
    },
}


async def scaffold_project(stack: str, project_name: str, output_dir: str = "") -> str:
    """Generate a project skeleton for the given stack.

    Returns a summary of created files.
    """
    if stack not in STACKS:
        available = ", ".join(STACKS.keys())
        return f"Unknown stack: '{stack}'. Available: {available}"

    template = STACKS[stack]
    if not output_dir:
        output_dir = project_name.lower().replace(" ", "_")

    created = []
    for rel_path, content in template["files"].items():
        full_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Replace template variables
        rendered = content.replace("{project_name}", project_name)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(rendered)
        created.append(rel_path)

    summary = f"Scaffolded '{stack}' project '{project_name}' at {output_dir}/\n"
    summary += f"Stack: {template['description']}\n"
    summary += f"Files created ({len(created)}):\n"
    for f in created:
        summary += f"  - {f}\n"

    return summary


def list_stacks() -> str:
    """List available project scaffolding stacks."""
    lines = ["Available stacks:"]
    for name, info in STACKS.items():
        lines.append(f"  {name}: {info['description']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stack recommendation engine
# ---------------------------------------------------------------------------

STACK_RECOMMENDATIONS = {
    "web_app": {
        "default": "nextjs",
        "with_api": "fastapi",
        "fullstack": ["nextjs", "fastapi"],
        "description": "Web application",
    },
    "mobile_app": {
        "default": "expo",
        "cross_platform": "expo",
        "description": "Mobile application (iOS/Android)",
    },
    "api_service": {
        "default": "fastapi",
        "lightweight": "flask",
        "description": "Backend API service",
    },
    "saas": {
        "default": ["nextjs", "fastapi"],
        "description": "SaaS product (frontend + backend)",
    },
}


def recommend_stack(project_type: str = "", requirements: str = "") -> str:
    """Recommend a tech stack based on project type and requirements."""
    if not project_type:
        lines = ["Project types and recommended stacks:"]
        for ptype, info in STACK_RECOMMENDATIONS.items():
            lines.append(f"  {ptype}: {info['description']} -> {info['default']}")
        return "\n".join(lines)

    ptype_lower = project_type.lower().replace(" ", "_")
    rec = STACK_RECOMMENDATIONS.get(ptype_lower)
    if not rec:
        # Try fuzzy match
        for key in STACK_RECOMMENDATIONS:
            if key in ptype_lower or ptype_lower in key:
                rec = STACK_RECOMMENDATIONS[key]
                break

    if not rec:
        return f"Unknown project type: '{project_type}'. Available: {', '.join(STACK_RECOMMENDATIONS.keys())}"

    stack = rec["default"]
    if isinstance(stack, list):
        stack_str = " + ".join(stack)
    else:
        stack_str = stack

    result = f"Recommended stack for {project_type}: {stack_str}\n"
    if isinstance(stack, list):
        for s in stack:
            if s in STACKS:
                result += f"\n{s}: {STACKS[s]['description']}"
    elif stack in STACKS:
        result += f"\n{STACKS[stack]['description']}"

    return result
