from config import OPENAI_API_KEY
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit 
from langchain. tools.playwright.utils import create_async_playwright_browser 
import asyncio

async_browser = create_async_playwright_browser()
browser_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
tools = browser_toolkit.get_tools()

#gpt_model = "gpt-4"

llm = ChatOpenAI(
    temperature=0,
    model = "gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY
)

agent_chain = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

async def main():
    response = await agent_chain.arun(input="Browse to https://www.theverge.com/2023/3/14/23638033/openai-gpt-4-chatgpt-multimodal-deep-learning and describe it to me.")
    return response

result = asyncio.run(main())
print(result)