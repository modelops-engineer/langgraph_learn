from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class Batsman(TypedDict):
    runs  : int
    balls : int
    fours : int
    sixes : int

    sr : float
    bpb : float
    bound_percent : float
    summary : str

def calc_sr(state : Batsman) -> Batsman:
    runs = state['runs']
    balls = state['balls']

    sr = (runs/balls)*100

    return {'sr':sr}

def calc_bpb(state : Batsman) -> Batsman:
    balls = state['balls']
    fours = state['fours']
    sixes = state['sixes']

    bpb = balls/(sixes + fours)

    return {'bpb':bpb}

def calc_bound_percent(state: Batsman) -> Batsman:
    runs = state['runs']
    fours = state['fours']
    sixes = state['sixes']

    bound_perc = ((fours*4 + sixes*6)/runs) * 100

    return {'bound_percent':bound_perc}

def summary(state: Batsman) -> Batsman:
    load_dotenv()
    model = ChatOpenAI()
    parser = StrOutputParser()

    runs = state['runs']
    balls = state['balls']
    fours = state['fours']
    sixes = state['sixes']
    sr = state['sr']
    bpb = state['bpb']
    bound_percent = state['bound_percent']


    prompt = PromptTemplate(
        template="""
        Provide a detailed summary for the batsman stats and also create a story for his performance.
        runs scored {runs}, strike rate {sr}, number of fours {fours}, number of sixes {sixes},
        balls per boundary {bpb}, balls faced {balls} and boundary percentage {bound_percent}
    """,
        input_variables=[
            'runs',
            'balls',
            'fours',
            'sixes',
            'sr', 
            'bpb',
            'bound_percent',
            'summary'
        ]
    )
    chain = prompt | model | parser

    result = chain.invoke(state)
    return {'summary' : result}


graph = StateGraph(Batsman)

# nodes
graph.add_node('calc_sr', calc_sr)
graph.add_node('calc_bpb', calc_bpb)
graph.add_node('calc_bound_percent', calc_bound_percent)
graph.add_node('summary', summary)


# edges
graph.add_edge(START, 'calc_sr')
graph.add_edge(START, 'calc_bpb')
graph.add_edge(START, 'calc_bound_percent')
graph.add_edge('calc_sr', 'summary')
graph.add_edge('calc_bpb', 'summary')
graph.add_edge('calc_bound_percent', 'summary')
graph.add_edge('summary', END)

# compile
workflow = graph.compile()

# execution
inital_input = {
    'runs' : 103,
    'balls' : 76,
    'fours' : 9,
    'sixes' : 7
}

ouput_state = workflow.invoke(inital_input)

print(ouput_state)