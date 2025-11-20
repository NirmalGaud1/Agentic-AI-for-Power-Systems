import streamlit as st
import sys
from io import StringIO

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.language_models import BaseLanguageModel
from typing import Any, Mapping, Optional, List

LOCAL_MODELS = {
    "Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "Phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
    "Gemma-2b-it": "google/gemma-2b-it",
    "Zephyr-7B-beta": "HuggingFaceH4/zephyr-7b-beta",
}
MODEL_IDS = list(LOCAL_MODELS.keys())

@tool
def power_system_calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Calculation Result: {result}"
    except Exception as e:
        return f"Error in calculation: {e}"

@tool
def retrieve_standard_document(topic: str) -> str:
    if "grounding resistance" in topic.lower():
        return "The typical maximum acceptable grounding resistance for a major substation is 5 Ohms, as per IEEE 80 standard guidelines."
    elif "feeder voltage level" in topic.lower():
        return "Standard distribution feeder voltages in the US are commonly 12.47 kV and 34.5 kV."
    else:
        return f"No specific standard found for the topic: {topic}. Try a different query."

tools = [power_system_calculator, retrieve_standard_document]

class MockLocalLLM(BaseLanguageModel):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if "power factor" in prompt.lower() or "calculate" in prompt.lower():
            return 'Thought: I need to calculate power.\nAction: power_system_calculator\nAction Input: 480 * 100 * 0.85 * (3**0.5) / 1000'
        elif "grounding" in prompt.lower() or "standard" in prompt.lower():
            return 'Thought: Need standard value.\nAction: retrieve_standard_document\nAction Input: grounding resistance'
        return "Thought: No tool needed.\nFinal Answer: Agentic AI uses reasoning and tools to solve complex tasks."

    @property
    def _llm_type(self) -> str:
        return "mock_local_llm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name": "MockLocalLLM"}

    def bind_tools(self, *args, **kwargs):
        return self

@st.cache_resource
def get_agent(model_key: str):
    st.subheader("Agent Setup Status")
    st.info(f"Selected Model: **{model_key}**")
    llm = MockLocalLLM()
    st.success("Mock Local LLM active (for demo)")

    prompt = PromptTemplate.from_template(
        """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    st.success("Agentic Engineer Initialized (ReAct)")
    return agent_executor

def capture_agent_output(agent_executor, goal):
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        response = agent_executor.invoke({"input": goal})
        verbose = sys.stdout.getvalue()
        return verbose, response["output"]
    finally:
        sys.stdout = old_stdout

def main():
    st.set_page_config(page_title="Agentic AI Engineer Demo", layout="wide")
    st.title("Agentic AI System for Power Engineering Demo")
    st.caption("Planning • Tool Use • Reflection")
    st.markdown("---")

    with st.sidebar:
        st.header("Model")
        selected_model = st.selectbox("Select Model", MODEL_IDS, index=0)

    agent_executor = get_agent(selected_model)

    st.header("Input Goal")
    prompt = st.text_input(
        "Enter engineering task:",
        value="A 480V three-phase motor draws 100 Amps at 0.85 power factor. Calculate real power in kW.",
        placeholder="e.g., What is the maximum allowed grounding resistance for a substation?"
    )

    if st.button("Run Agent"):
        if not prompt:
            st.error("Enter a goal")
            return
        with st.spinner("Agent thinking..."):
            verbose, answer = capture_agent_output(agent_executor, prompt)

        st.markdown("---")
        st.header("Agent Workflow")
        st.code(verbose, language="text")

        st.markdown("---")
        st.header("Final Answer")
        st.balloons()
        st.success(answer)

if __name__ == "__main__":
    main()
