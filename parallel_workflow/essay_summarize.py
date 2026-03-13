from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Literal, Optional, TypedDict, Annotated
import operator
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini')

class EvaluationSchema(BaseModel):

    feedback : str = Field(description='Detailed feedback for the essay')
    score : int = Field(description='Score for the essay out of 10', ge=0,le=10)


structured_model = model.with_structured_output(EvaluationSchema)

essay = """
Essay Topic: Artificial Intelligence and the Transformation of the Modern World

Artificial Intelligence (AI), once confined to the realm of science fiction, has rapidly evolved into a transformative force shaping the modern world. From healthcare and governance to economics and geopolitics, AI is redefining how societies function, how economies grow, and how individuals interact with technology. As nations compete to harness its potential, AI has emerged not merely as a technological innovation but as a structural force that is altering the foundations of global development.

At its core, AI refers to the ability of machines to perform tasks that typically require human intelligence—such as learning, reasoning, perception, and decision-making. The convergence of big data, powerful computing infrastructure, and advanced algorithms has accelerated AI development in the past decade. This convergence has enabled machines to recognize patterns in massive datasets, automate complex processes, and even generate creative outputs. As a result, AI has transitioned from experimental laboratories to everyday applications across industries.

One of the most profound impacts of AI is visible in the economic sphere. AI-driven automation is transforming production systems, enhancing efficiency, and reducing operational costs. Industries such as manufacturing, logistics, finance, and retail increasingly rely on intelligent algorithms to optimize supply chains, detect fraud, predict demand, and personalize consumer experiences. According to several economic analyses, AI could contribute trillions of dollars to the global economy in the coming decades. However, this technological shift also raises concerns about job displacement, widening inequality, and the need for large-scale reskilling of the workforce. Thus, AI presents both opportunities for economic growth and challenges for inclusive development.

In the domain of healthcare, AI is revolutionizing diagnostics, treatment planning, and drug discovery. Machine learning models can analyze medical images with remarkable accuracy, assisting doctors in detecting diseases such as cancer at early stages. AI-powered predictive analytics helps identify potential health risks and enables personalized treatment strategies. During global health crises, AI has also played a role in tracking disease spread and accelerating vaccine research. These developments demonstrate AI’s potential to strengthen healthcare systems and improve human well-being, particularly when integrated with robust public health policies.

Governance and public administration are also undergoing transformation due to AI. Governments across the world are exploring AI-driven solutions for improving public service delivery, enhancing policy analysis, and strengthening administrative efficiency. AI-based systems can assist in traffic management, urban planning, disaster prediction, and law enforcement. For a country like India, AI holds immense potential in addressing developmental challenges such as agricultural productivity, financial inclusion, and education accessibility. However, the deployment of AI in governance must be accompanied by safeguards to protect privacy, prevent algorithmic bias, and ensure transparency in decision-making.

Beyond economic and administrative domains, AI is reshaping global geopolitics. Nations increasingly view AI as a strategic asset that could determine technological and military superiority in the twenty-first century. Investments in AI research, semiconductor technology, and advanced computing infrastructure have become central to national security strategies. The race for AI dominance among major powers is likely to influence international relations, trade policies, and technological alliances. Consequently, global cooperation and ethical frameworks will be essential to ensure that AI development benefits humanity rather than exacerbating geopolitical tensions.

Despite its immense potential, AI raises complex ethical and societal questions. Issues related to data privacy, surveillance, algorithmic discrimination, and accountability have become prominent concerns. AI systems trained on biased datasets can perpetuate social inequalities, while unchecked surveillance technologies may threaten civil liberties. Furthermore, the increasing autonomy of AI systems raises questions about responsibility and regulation. Policymakers, technologists, and civil society must therefore collaborate to develop ethical guidelines and regulatory frameworks that ensure responsible AI development.

Education and human capital development will play a crucial role in navigating the AI-driven future. As routine tasks become automated, the demand for skills such as critical thinking, creativity, interdisciplinary knowledge, and technological literacy will increase. Educational systems must adapt to prepare individuals for an evolving job market shaped by intelligent technologies. Lifelong learning and reskilling initiatives will be essential to ensure that societies can harness AI's benefits while minimizing its disruptive effects.

In conclusion, Artificial Intelligence represents one of the most consequential technological transformations of the modern era. Its influence extends across economic systems, governance structures, healthcare, and global geopolitics. While AI offers unprecedented opportunities to enhance productivity, improve public services, and address complex global challenges, it also poses significant ethical, social, and regulatory dilemmas. The future of AI will ultimately depend on how societies choose to govern and integrate this powerful technology. If guided by principles of inclusivity, transparency, and human welfare, AI has the potential to become a transformative force that advances collective progress and reshapes the world for the better.
"""

essay2 = """
Artificial Intelligence Changing the World

Artificial Intelligence is a technology which is changing the world very fast in today’s modern era. Artificial Intelligence is called AI and it is very important technology nowadays. Many countries are using AI for development and improvement of their systems. Because of this reason AI is becoming more popular day by day.

AI is used in many places like hospitals, companies, schools and many other industries. In hospitals AI helps doctors to detect diseases. In companies AI helps to do work faster and improve productivity. AI is also used in mobile phones, computers and other electronic devices. Because of these reasons AI is becoming necessary for everyone.

However AI also has some disadvantages. AI can reduce jobs because machines can do work which humans used to do earlier. This can create unemployment problems in society. Also AI systems depend on data and sometimes the data may not be correct. Because of this AI decisions can also become wrong.

Another important point is that many countries are investing in AI research. They want to become leaders in AI technology. This can increase competition between nations. Governments should make rules and policies to control AI so that it is used properly.

In conclusion, Artificial Intelligence is a very useful technology which is changing the world rapidly. It has many advantages and some disadvantages. If AI is used carefully and responsibly then it can help humanity to progress in future.
"""

class EssayState(TypedDict):

    essay : str
    language_feedback : str
    analysis_feedback : str
    clarity_feedback : str
    overall_feedback : str
    individual_score : Annotated[list[int], operator.add]
    average_score : float



graph = StateGraph(EssayState)


def evaluate_language(state : EssayState) -> EssayState:

    essay = state['essay']
    prompt = f'Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n essay : \n {essay}'

    output = structured_model.invoke(prompt)

    return {'language_feedback':output.feedback, 'individual_score':[output.score]}

def evaluate_analysis(state: EssayState) -> EssayState:

    essay = state['essay']
    prompt = f'Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10 \n essay: \n {essay}'

    output = structured_model.invoke(prompt)

    return {'analysis_feedback':output.feedback, 'individual_score': [output.score]}

def evaluate_clarity(state: EssayState) -> EssayState:

    essay = state['essay']
    prompt = f'Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10 \n essay: \n {essay}'

    output = structured_model.invoke(prompt)

    return {'clarity_feedback': output.feedback, 'individual_score':[output.score]}

def final_evaluation(state: EssayState) -> EssayState:

    prompt = f'Based on the following feedbacks create a summarized feedback \n language feedback : \n {state["language_feedback"]} \n depth of analysis feedback : \n {state["analysis_feedback"]} \n clarity of thought feedback :\n {state["clarity_feedback"]}'
    overall_feedback = model.invoke(prompt).content

    avg_score = sum(state['individual_score'])/len(state['individual_score'])

    return {'overall_feedback' : overall_feedback, 'average_score':avg_score}

# add node
node1 = graph.add_node('evaluate_language', evaluate_language)
node1 = graph.add_node('evaluate_analysis', evaluate_analysis)
node1 = graph.add_node('evaluate_clarity', evaluate_clarity)
node1 = graph.add_node('final_evaluation', final_evaluation)

# add edge
edge1 = graph.add_edge(START, 'evaluate_language')
edge2 = graph.add_edge(START, 'evaluate_analysis')
edge3 = graph.add_edge(START, 'evaluate_clarity')

edge4 = graph.add_edge('evaluate_language', 'final_evaluation')
edge5 = graph.add_edge('evaluate_analysis', 'final_evaluation')
edge6 = graph.add_edge('evaluate_clarity', 'final_evaluation')

edge7 = graph.add_edge('final_evaluation', END)


# compilation

workflow = graph.compile()

# png_bytes = workflow.get_graph().draw_mermaid_png()
# with open("essay_graph.png", "wb") as f:
#     f.write(png_bytes)

initial_state = {
    'essay' : essay2
}

result = workflow.invoke(initial_state)

print(result)