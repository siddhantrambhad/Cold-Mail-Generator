import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Siddhant Rambhad, a skilled developer and technology enthusiast with expertise in full-stack web development, 
real-time applications, and machine learning. You specialize in creating efficient, scalable, and user-focused solutions 
using technologies like React.js, Next.js, Node.js, Python, Tailwind CSS, and computer vision frameworks.

Your role is to write a professional cold email to the client regarding the job mentioned above, 
explaining how your technical expertise and project experience can fulfill their requirements.

Highlight relevant work from your portfolio — including:
• We Meet a real-time video conferencing platform.
• Grilli Restaurant Website a responsive, user-friendly web application.
• YOLOv3 Object Detection  a low-latency computer vision system.

Also, include the most relevant ones from the following links to strengthen your proposal: {link_list}  
Do not provide a preamble.  

### EMAIL (NO PREAMBLE):
"""
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))