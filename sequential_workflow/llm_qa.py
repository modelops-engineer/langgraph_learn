from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()



class LLMState(TypedDict):
    quest:str
    ans:str

graph = StateGraph(LLMState)

def llm_qa(state:LLMState) -> LLMState:
    # extract the question
    quest = state['quest']

    # prompt
    prompt = PromptTemplate(
    template='''
    Answer the following {question} in a shot paragraph, but if it is a factual question just answer in 1 line do not elongate.
    ''',
    input_variables=['question']
    )

    chain = prompt | model | parser
    state['ans'] = chain.invoke({'question':quest})

    return state

# nodes
node1 = graph.add_node('llm_qa', llm_qa)


# edges
edge1 = graph.add_edge(START, 'llm_qa')
edge2 = graph.add_edge('llm_qa', END)


# compile
workflow = graph.compile()


# execute
initial_question = {
    'quest' : "What is the requirement of Langchain in Langgraph"
}
output = workflow.invoke(initial_question)
print(output)