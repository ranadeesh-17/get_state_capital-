import os
import io
import sys
import traceback

from typing import TypedDict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END


# ============================================================
# GEMINI API KEY
# ============================================================

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not configured.")


# ============================================================
# GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=API_KEY,
    temperature=0
)


# ============================================================
# LANGGRAPH STATE
# ============================================================

class CrewState(TypedDict, total=False):
    messages: List[BaseMessage]
    code: Optional[str]
    report: Optional[str]


# ============================================================
# TOOL: RUN PYTHON CODE
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return output or error."""

    clean_code = (
        str(code)
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:
        exec(clean_code, {}, {})
        result = new_stdout.getvalue()

    except Exception:
        result = traceback.format_exc()

    finally:
        sys.stdout = old_stdout

    return result.strip() or "Success (no terminal output)"


# ============================================================
# TOOL: GENERATE TEST CASES
# ============================================================

@tool
def generate_test_cases(task_description: str) -> str:
    """Generate test cases for a coding task."""

    prompt = f"""
You are a senior QA engineer.

Generate 3 to 5 specific test cases for this Python programming task:

{task_description}

Include:
1. Normal cases
2. Edge cases
3. Invalid cases

Return only a numbered list.
"""

    response = llm.invoke(prompt)

    return str(response.content)


# ============================================================
# DEVELOPER NODE
# ============================================================

def developer_node(state: CrewState):

    task = state["messages"][-1].content

    prompt = f"""
You are a senior Python developer.

Write a clean Python program to solve this task:

{task}

Rules:
- Return ONLY Python code.
- Do not use Markdown.
- Do not include ```python.
- Do not explain the code.
- The program must be executable.
- Use standard Python libraries whenever possible.
"""

    response = llm.invoke(prompt)

    code = str(response.content)

    code = (
        code
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    return {
        "code": code
    }


# ============================================================
# TESTER NODE
# ============================================================

def tester_node(state: CrewState):

    task = state["messages"][-1].content

    tests = generate_test_cases.invoke({
        "task_description": task
    })

    execution = run_python_code.invoke({
        "code": state["code"]
    })

    report = f"""
GENERATED CODE

{state["code"]}


EXECUTION OUTPUT

{execution}


TEST CASES

{tests}
"""

    return {
        "report": report
    }


# ============================================================
# LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(CrewState)

workflow.add_node(
    "developer",
    developer_node
)

workflow.add_node(
    "tester",
    tester_node
)

workflow.add_edge(
    START,
    "developer"
)

workflow.add_edge(
    "developer",
    "tester"
)

workflow.add_edge(
    "tester",
    END
)

graph = workflow.compile()


# ============================================================
# RUN LANGGRAPH
# ============================================================

def run_langgraph(task: str):

    result = graph.invoke({
        "messages": [
            HumanMessage(content=task)
        ],
        "code": None,
        "report": None
    })

    return result["report"]


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="LangGraph Coding Agent",
    version="1.0.0",

    # Disable Swagger
    docs_url=None,

    # Disable ReDoc
    redoc_url=None,

    # Disable OpenAPI
    openapi_url=None
)


# ============================================================
# LANGGRAPH WEB PAGE
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>

<html>

<head>

<title>LangGraph Coding Agent</title>

<meta name="viewport"
      content="width=device-width, initial-scale=1">

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f3f4f6;
}

.container {
    max-width: 900px;
    margin: 50px auto;
    padding: 20px;
}

.card {
    background: white;
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0 5px 25px rgba(0,0,0,0.08);
}

h1 {
    text-align: center;
    margin-top: 0;
}

.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 25px;
}

textarea {
    width: 100%;
    min-height: 180px;
    padding: 15px;
    font-size: 16px;
    border: 1px solid #ccc;
    border-radius: 10px;
    resize: vertical;
}

button {
    width: 100%;
    margin-top: 15px;
    padding: 14px;
    border: none;
    border-radius: 10px;
    background: #111827;
    color: white;
    font-size: 16px;
    cursor: pointer;
}

button:hover {
    background: #000;
}

button:disabled {
    background: #888;
    cursor: not-allowed;
}

#loading {
    display: none;
    text-align: center;
    margin-top: 15px;
    color: #555;
}

#result {
    margin-top: 25px;
    padding: 20px;
    background: #f8fafc;
    border: 1px solid #ddd;
    border-radius: 10px;
    white-space: pre-wrap;
    overflow-x: auto;
    font-family: Consolas, monospace;
}

</style>

</head>


<body>

<div class="container">

<div class="card">

<h1>LangGraph Coding Agent</h1>

<div class="subtitle">
Enter a Python programming task
</div>

<textarea
    id="task"
    placeholder="Example: Write a Python program to calculate factorial">
</textarea>

<button
    id="runButton"
    onclick="runLangGraph()">

    Run LangGraph

</button>

<div id="loading">
    Running LangGraph...
</div>

<div id="result">
    Result will appear here.
</div>

</div>

</div>


<script>

async function runLangGraph() {

    const task =
        document.getElementById("task").value.trim();

    const result =
        document.getElementById("result");

    const button =
        document.getElementById("runButton");

    const loading =
        document.getElementById("loading");


    if (!task) {

        result.textContent =
            "Please enter a coding task.";

        return;
    }


    button.disabled = true;

    loading.style.display = "block";

    result.textContent = "Processing...";


    try {

        const response = await fetch(
            "/langgraph",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    task: task
                })
            }
        );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Something went wrong."
            );
        }


        result.textContent =
            data.result;


    } catch (error) {

        result.textContent =
            "Error: " + error.message;

    }


    button.disabled = false;

    loading.style.display = "none";
}

</script>

</body>

</html>
"""


# ============================================================
# GET /langgraph
# ============================================================

@app.get(
    "/langgraph",
    response_class=HTMLResponse
)
def langgraph_page():

    return HTML_PAGE


# ============================================================
# POST /langgraph
# ============================================================

@app.post("/langgraph")
async def langgraph_endpoint(request: dict):

    task = request.get("task")

    if not task:
        raise HTTPException(
            status_code=400,
            detail="Task is required."
        )

    try:

        print(
            f"[LangGraph] Received task: {task}"
        )

        result = run_langgraph(task)

        print(
            "[LangGraph] Execution completed."
        )

        return {
            "success": True,
            "result": result
        }

    except Exception as e:

        print(
            f"[LangGraph] ERROR: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "LangGraph Coding Agent"
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "application": "LangGraph Coding Agent",
        "open": "/langgraph"
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
