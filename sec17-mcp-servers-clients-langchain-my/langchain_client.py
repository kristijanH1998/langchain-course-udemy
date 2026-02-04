from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import asyncio
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatOpenAI()

stdio_server_params = StdioServerParameters(
    command="python",
    args=["/home/kiki/git/langchain-course/sec17-mcp-servers-clients-langchain/mcp-crash-course/servers/math_server.py"],
)

async def main():
    print("hello langchain mcp")
    async with stdio_client(stdio_server_params) as (read,write):
        async with ClientSession(read_stream=read, write_stream=write) as session:
            await session.initialize()
            print("session initialized")    
            # tools = await session.list_tools()
            # print(tools)
            tools = await load_mcp_tools(session)
            agent = create_react_agent(llm, tools)
            # print(tools)
            result = await agent.ainvoke({"messages": [HumanMessage(content="What is 2 + 2?")]})
            print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())