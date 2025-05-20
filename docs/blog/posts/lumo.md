---
draft: false 
date: 2025-05-01
authors: 
 - sonam
slug: mcp, agent
title: Easy MCP integration to our agentic framework; LUMO
---

## Building a Server with Lumo: A Step-by-Step Guide to MCP Integration
Lumo, a powerful Rust-based agent, offers seamless integration with MCPs (Modular Control Protocols) and remarkable flexibility in implementation. While Lumo can be used as a library, CLI tool, or server, this guide will focus specifically on deploying Lumo in server mode for optimal MCP integration.
<!-- more -->

## What is an MCP?
Modular Control Protocol (MCP) is a standardized communication framework that allows different components of a system to interact efficiently. MCPs enable modular applications to communicate through a structured protocol, making it easier to build scalable, maintainable systems where components can be swapped or upgraded without disrupting the entire architecture.

## Architecture of MCP

MCP follows a client-server architecture with clearly defined roles:

- **Hosts**: LLM applications (like Claude Desktop or integrated development environments) that initiate connections
- **Clients**: Components that maintain one-to-one connections with servers inside the host application
- **Servers**: Systems that provide context, tools, and prompts to clients

This architecture is built around three main concepts:

1. **Resources**: Similar to GET endpoints, resources load information into the LLM's context
2. **Tools**: Functioning like POST endpoints, tools execute code or produce side effects
3. **Prompts**: Reusable templates that define interaction patterns for LLM communications

![alt text](https://royal-hygienic-522.notion.site/image/attachment%3Ac462d75f-ac1f-460b-b686-8bd3827a4f6d%3Aimage.png?table=block&id=1f981b6a-6bbe-805d-8ce5-e6b1bf4697ce&spaceId=f1bf59bf-2c3f-4b4d-a5f9-109d041ef45a&width=1270&userId=&cache=v2)

## Setting Up a Lumo Server with MCP Integration

## 🖥️ Server Usage

Lumo can also be run as a server, providing a REST API for agent interactions.

### Starting the Server

```
cargo install --git https://github.com/StarlightSearch/lumo.git --branch new-updates --features mcp lumo-server

```

#### Using Binary
```bash
# Start the server (default port: 8080)
lumo-server
```

#### Using Docker
```bash
# Build the image
docker build -f server.Dockerfile -t lumo-server .

# Run the container with required API keys
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=your-openai-key \
  -e GOOGLE_API_KEY=your-google-key \
  -e GROQ_API_KEY=your-groq-key \
  -e ANTHROPIC_API_KEY=your-anthropic-key \
  -e EXA_API_KEY=your-exa-key \
  lumo-server
```

You can also use the pre-built image:
```bash
docker pull akshayballal95/lumo-server:latest
```

### Server Configuration
You can configure multiple servers in the configuration file for MCP agent usage. The configuration file location varies by operating system:

```
Linux: ~/.config/lumo-cli/servers.yaml
macOS: ~/Library/Application Support/lumo-cli/servers.yaml
Windows: %APPDATA%\Roaming\lumo\lumo-cli\servers.yaml```

Example config: 

```exa-search:
  command: npx
  args:
    - "exa-mcp-server"
  env: 
    EXA_API_KEY: "your-api-key"

fetch:
  command: uvx
  args:
    - "mcp_server_fetch"

system_prompt: |-
  You are a powerful agentic AI assistant...

```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8080/health_check
```

#### Run Task
```bash
curl -X POST http://localhost:8080/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "What are the files in the folder?",
    "model": "gpt-4o-mini",
    "base_url": "https://api.openai.com/v1/chat/completions",
    "tools": ["DuckDuckGo", "VisitWebsite"],
    "max_steps": 5,
    "agent_type": "mcp"
  }'
```

#### Request Body Parameters

- `task` (required): The task to execute
- `model` (required): Model ID (e.g., "gpt-4", "qwen2.5", "gemini-2.0-flash")
- `base_url` (required): Base URL for the API
- `tools` (optional): Array of tool names to use
- `max_steps` (optional): Maximum number of steps to take
- `agent_type` (optional): Type of agent to use ("function-calling" or "mcp")
- `history` (optional): Array of previous messages for context



---
## MCP vs. Traditional Function-Calling

While both MCP and traditional function-calling allow LLMs to interact with external tools, they differ in several important ways:

The only difference between them is that, in traditional function calling, you need to define the processes and then LLM chooses which is the right option for the given job. It’s main purpose is to translate natural language into JSON format function calls. Meanwhile, MCP is the protocol that standardized the resources and tool calls for the LLM, that is why even though LLM still makes the decision to choose which MCP, it’s the standard calls that makes it highly scalable

### Benefits of lumo over other agentic systems

1. MCP Agent Support for multi-agent coordination
2. Multi-modal support, can easily use OpenAI, Google or Anthropic
3. Asynchronous tool calling.
4. In-built Observability with langfuse

Open discussions if you have any doubt and give us a star at repo.

## Conclusion

As agents evolve, standardized protocols like MCP will become increasingly important for enabling sophisticated AI applications. By providing a common language for AI systems to interact with external tools and data sources, MCP helps bridge the gap between powerful language models and the specific capabilities needed for real-world applications.

For developers working with AI, understanding and adopting MCP offers a more sustainable, future-proof approach to building AI integrations compared to platform-specific function-calling implementations.