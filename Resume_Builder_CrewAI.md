!pip install crewai crewai-tools

import os
from crewai import Agent, Task, Crew
from crewai_tools import (
  FileReadTool,
  ScrapeWebsiteTool,
  PDFSearchTool,
  SerperDevTool
)
from crewai import LLM


os.environ["SERPER_API_KEY"] = "d9ff48be72da913afbbd365097415340c0aedc7b"
os.environ['MISTRAL_API_KEY'] = "uYrvStjJisQDYiMejb5Ib4jbVt7OMrzd"


# Initialize the semantic search tool for resume search using Mistral AI and Google embeddings
semantic_search_resume = PDFSearchTool(
    config=dict(
        # Configure the language model to use Mistral AI as the provider
        llm=dict(
            provider="mistralai",  # Change to use Mistral AI as the provider
            config=dict(
                model="mistral-7B",  # Specify the Mistral AI model to use
            ),
        ),
        # Configure the embedding model to use Google's document retrieval embedding model
        embedder=dict(
            provider="google",  # You can change this to other providers like openai, ollama, etc.
            config=dict(
                model="models/embedding-001",  # Specify the embedding model for retrieval tasks
                task_type="retrieval_document",  # Define the task type as document retrieval
            ),
        ),
    ),
    # Specify the path to the resume PDF for semantic search
    mdx="/content/Sami_Thakur(M).pdf"  # Path to the resume PDF file
)


# Initialize the search tool using SerperDev for semantic search capabilities
search_tool = SerperDevTool(
      config=dict(
        engine="google_scholar",  # Use Google Scholar for research-oriented searches
        num_results=10,          # Get more results
    )
)

# Initialize the web scraping tool to extract content from websites
scrape_tool = ScrapeWebsiteTool()

# Initialize the file reading tool to read the resume PDF file for further analysis
read_resume = FileReadTool(file_path="/content/Sami_Thakur(M).pdf")

mistral = LLM(
    model="mistral-small-2402",  # Specify the Mistral model to use
    temperature=0.1,  # Adjust the creativity level of the model's responses
    base_url="https://api.mistral.ai/v1",  # Base URL for the Mistral API
    api_key="uYrvStjJisQDYiMejb5Ib4jbVt7OMrzd"  # Your Mistral API key (replace as needed)
)
gemini = LLM(
    model="gemini/gemini-pro",  # Specify the Gemini model
    temperature=0.1,  # Control the level of variability in the model's output
    api_key="AIzaSyAWi-8trRZjEXnDiqDfzZeWLabWVjGeMFs"  # Your Gemini API key (replace as needed)
)


researcher = Agent(
    role="Tech Job Researcher",  # Define the role of the agent
    goal="Make sure to do amazing analysis on "
         "job posting to help job applicants",  # Set the goal for the agent
    tools=[scrape_tool, search_tool],  # Assign the necessary tools for research (scraping and search)
    verbose=True,  # Enable verbose mode to print detailed process logs
    backstory=(
        "As a Job Researcher, your prowess in "
        "navigating and extracting critical "
        "information from job postings is unmatched."
        "Your skills help pinpoint the necessary "
        "qualifications and skills sought "
        "by employers, forming the foundation for "
        "effective application tailoring."  # Backstory for the agent's expertise
    ),
    llm=gemini
)


#Initialize the second agent, 'Personal Profiler for Engineers', for applicant profiling
profiler = Agent(
    role="Personal Profiler for Engineers",  # Define the agent's role as a profiler for engineers
    goal="Do incredible research on job applicants "
         "to help them stand out in the job market",  # Set the goal to improve applicants' visibility in the job market
    tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],  # Assign tools for scraping, searching, and resume analysis
    verbose=True,  # Enable verbose mode for detailed output logs
    backstory=(
        "Equipped with analytical prowess, you dissect "
        "and synthesize information "
        "from diverse sources to craft comprehensive "
        "personal and professional profiles, laying the "
        "groundwork for personalized resume enhancements."  # Backstory describing the agent's expertise in profiling
    ),
    llm=mistral
)


# Initialize the third agent, 'Resume Strategist for Engineers', for resume optimization
resume_strategist = Agent(
    role="Resume Strategist for Software Engineers",  # Define the agent's role focused on resume strategy
    goal="Find all the best ways to make a "
         "resume stand out in the job market.",  # Set the goal to optimize resumes for better visibility
    tools=[read_resume, semantic_search_resume],  # Assign tools for resume analysis, web scraping, and job searching
    verbose=True,  # Enable verbose mode for detailed logging of the agent's actions
    backstory=(
        "With a strategic mind and an eye for detail, you "
        "excel at refining resumes to highlight the most "
        "relevant skills and experiences, ensuring they "
        "resonate perfectly with the job's requirements."  # Backstory highlighting the agent's expertise in resume optimization
    ),
    llm=mistral
)


# Initialize the fourth agent, 'Engineering Interview Preparer', for interview preparation
interview_preparer = Agent(
    role="Engineering Interview Preparer",  # Define the agent's role as an interview preparer
    goal="Create interview questions and talking points "
         "based on the resume and job requirements",  # Set the goal to help candidates prepare for interviews
    tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],  # Assign tools for scraping, searching, and resume analysis
    verbose=True,  # Enable verbose mode to print detailed logs of the agent's actions
    backstory=(
        "Your role is crucial in anticipating the dynamics of "
        "interviews. With your ability to formulate key questions "
        "and talking points, you prepare candidates for success, "
        "ensuring they can confidently address all aspects of the "
        "job they are applying for."  # Backstory highlighting the agent's expertise in interview preparation
    ),
    llm=gemini
)


# Task for Researcher Agent: Extract Job Requirements
research_task = Task(
    description=(
        "Analyze the job posting URL provided ({job_posting_url}) "
        "to extract key skills, experiences, and qualifications "
        "required. Use the tools to gather content and identify "
        "and categorize the requirements."
    ),
    expected_output=(
        "A structured list of job requirements, including necessary "
        "skills, qualifications, and experiences."
    ),
    agent=researcher,
    async_execution=True
)


profile_task = Task(
    description=(
        "Compile a detailed personal and professional profile "  # Task description indicating the need for a profile compilation
        "using the GitHub ({github_url}) URLs, and personal write-up "
        "({personal_writeup}). Utilize tools to extract and "
        "synthesize information from these sources."  # Guide the agent to extract and synthesize information
    ),
    expected_output=(
        "A comprehensive profile document that includes skills, "  # Specify the expected output as a detailed profile
        "project experiences, contributions, interests, and "
        "communication style."  # Outline the key components to be included in the profile
    ),
    agent=profiler,  # Assign the Profiler Agent to complete the task
    async_execution=True  # Enable asynchronous execution to allow the agent to work in the background
)


# Task for Resume Strategist Agent: Align Resume with Job Requirements
resume_strategy_task = Task(
    description=(
        "Using **only** the profile and job requirements obtained from "
        "previous tasks, tailor the resume to highlight the most "
        "relevant areas. Employ tools to adjust and enhance the "
        "resume content. Make sure this is the best resume even but "
        "don't make up any information. Update every section, "
        "inlcuding the initial summary, work experience, skills, "
        "and education. All to better reflrect the candidates "
        "abilities and how it matches the job posting."
    ),
    expected_output=(
        "An updated resume that effectively highlights the candidate's "
        "qualifications and experiences relevant to the job."
    ),
    output_file="tailored_resume.md",
    context=[research_task, profile_task],
    agent=resume_strategist
)


# Task for Interview Preparer Agent: Develop Interview Materials
interview_preparation_task = Task(
    description=(
        "Create a list of 10 potential interview questions and talking "
        "points based on the tailored resume and job requirements. "
        "Utilize tools to generate relevant questions and discussion "
        "points. Make sure to use these question and talking points to "
        "help the candiadte highlight the main points of the resume "
        "and how it matches the job posting."
    ),
    expected_output=(
        "A document containing key questions and talking points "
        "that the candidate should prepare for the initial interview."
    ),
    output_file="interview_materials.md",
    context=[research_task, profile_task, resume_strategy_task],
    agent=interview_preparer
)


job_application_crew = Crew(
    agents=[researcher,
            profiler,
            resume_strategist,
            interview_preparer],

    tasks=[research_task,
           profile_task,
           resume_strategy_task,
           interview_preparation_task],

    verbose=True
)


job_application_inputs = {
    'job_posting_url': 'https://branchinternational.applytojob.com/apply/nT6J8qzm72/ML-Engineer-Intern?source=Our%20Career%20Page%20Widget',
    'github_url': 'https://github.com/SAMI-THAKUR',
    'personal_writeup': """Hello, my name is Sami. I am a third-year engineering student at VESIT (Mumbai), majoring in Artificial Intelligence and
Data Science. Over the past two years, I've developed a strong proficiency in Full Stack Web Development, having
completed several projects that showcase my skills in building comprehensive applications. I have a solid grasp of core
Java and am well-versed in machine learning algorithms. Currently, I am actively solving LeetCode problems, with a
rating of 1550, and am expanding my knowledge in Deep Learning and Generative AI.
"""

}


result = job_application_crew.kickoff(inputs=job_application_inputs)
