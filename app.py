import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda


# --- 1. Define Tool ---

@tool
def get_state_capital(state: str) -> str:
    """Get the capital city of an Indian state."""

    state_capitals = {
        "andhra pradesh": "Amaravati",
        "arunachal pradesh": "Itanagar",
        "assam": "Dispur",
        "bihar": "Patna",
        "chhattisgarh": "Raipur",
        "goa": "Panaji",
        "gujarat": "Gandhinagar",
        "haryana": "Chandigarh",
        "himachal pradesh": "Shimla",
        "jharkhand": "Ranchi",
        "karnataka": "Bengaluru",
        "kerala": "Thiruvananthapuram",
        "madhya pradesh": "Bhopal",
        "maharashtra": "Mumbai",
        "manipur": "Imphal",
        "meghalaya": "Shillong",
        "mizoram": "Aizawl",
        "nagaland": "Kohima",
        "odisha": "Bhubaneswar",
        "punjab": "Chandigarh",
        "rajasthan": "Jaipur",
        "sikkim": "Gangtok",
        "tamil nadu": "Chennai",
        "telangana": "Hyderabad",
        "tripura": "Agartala",
        "uttar pradesh": "Lucknow",
        "uttarakhand": "Dehradun",
        "west bengal": "Kolkata"
    }

    state_key = state.strip().lower()

    result = state_capitals.get(state_key)

    if result:
        return f"The capital of {state.title()} is {result}."

    return (
        f"Capital information not found for {state.title()}. "
        "Please provide a valid Indian state."
    )


tools = [
    get_state_capital
]


# --- 2. Initialize Model & Agent ---

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GOOGLE_API_KEY,
    temperature=0
)


agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a specialized agent restricted ONLY to Indian states "
        "and their state capitals. "
        "You can answer questions about the capital of an Indian state. "
        "For any other topics, questions, countries, currencies, "
        "or general knowledge outside of Indian states and their capitals, "
        "you must say exactly: "
        "'I am not authorized to answer questions outside of Indian states and their capitals.'"
    )
)


# --- 3. Input Model ---

class AgentInput(BaseModel):
    input: str = Field(description="Your message to the agent")


def format_for_agent(x) -> dict:

    user_input = (
        x["input"]
        if isinstance(x, dict)
        else x.input
    )

    return {
        "messages": [
            ("user", user_input)
        ]
    }


def extract_text_response(agent_output: dict) -> str:

    if not isinstance(agent_output, dict):
        return str(agent_output)

    # Case 1: top-level messages
    messages = agent_output.get("messages")

    # Case 2: nested under a node name
    if messages is None:

        for value in agent_output.values():

            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:

        last = messages[-1]

        return getattr(
            last,
            "content",
            str(last)
        )

    return str(agent_output)


formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# --- 4. FastAPI App ---

app = FastAPI(
    title="Indian State Capital Agent",
    version="1.0",
    description=(
        "A LangChain agent using Gemini with a tool "
        "for finding the capitals of Indian states."
    )
)


@app.get("/")
def root():

    return {
        "message": (
            "Server is running. "
            "Visit /agent/playground/ to chat, "
            "or /docs for the API."
        )
    }


# --- 5. Add LangServe Route ---

add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


# --- 6. Run Server ---

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
