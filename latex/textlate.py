import subprocess
import os
from pathlib import Path
import tempfile
import shutil


class LaTeXToPDF:
    def __init__(self, docker_image="texlive/texlive:latest", use_persistent_container=True):
        """
        Initialize LaTeX to PDF converter using Docker.

        Args:
            docker_image: Docker image to use (default: texlive/texlive:latest)
            use_persistent_container: Keep a long-running container (faster for repeated use)
        """
        self.docker_image = docker_image
        self.use_persistent_container = use_persistent_container
        self.container_name = "latex-compiler"

        self._ensure_docker_image()

        if self.use_persistent_container:
            self._ensure_persistent_container()

    def _ensure_docker_image(self):
        """Pull the Docker image if it doesn't exist."""
        try:
            result = subprocess.run(
                ["docker", "images", "-q", self.docker_image],
                capture_output=True,
                text=True,
                check=True
            )

            if not result.stdout.strip():
                print(f"Pulling Docker image {self.docker_image}...")
                subprocess.run(
                    ["docker", "pull", self.docker_image],
                    check=True
                )
                print("Image pulled successfully!")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Docker error: {e}")

    def _ensure_persistent_container(self):
        """Start a long-running container if it doesn't exist."""
        try:
            # Check if container exists and is running
            result = subprocess.run(
                ["docker", "ps", "-q", "-f", f"name={self.container_name}"],
                capture_output=True,
                text=True,
                check=True
            )

            if result.stdout.strip():
                print(f"Using existing container: {self.container_name}")
                return

            # Check if container exists but is stopped
            result = subprocess.run(
                ["docker", "ps", "-aq", "-f", f"name={self.container_name}"],
                capture_output=True,
                text=True,
                check=True
            )

            if result.stdout.strip():
                # Start existing container
                print(f"Starting existing container: {self.container_name}")
                subprocess.run(
                    ["docker", "start", self.container_name],
                    check=True
                )
            else:
                # Create and start new container
                print(f"Creating persistent container: {self.container_name}")
                subprocess.run(
                    [
                        "docker", "run",
                        "-d",  # Detached mode
                        "--name", self.container_name,
                        self.docker_image,
                        "tail", "-f", "/dev/null"  # Keep container alive
                    ],
                    check=True
                )
                print(f"Container {self.container_name} created and running!")

        except subprocess.CalledProcessError as e:
            raise Exception(f"Error managing persistent container: {e}")

    def compile_latex(self, latex_content, output_path=None, engine="pdflatex",
                      clean_aux=True, keep_temp=False):
        """
        Compile LaTeX content to PDF.

        Args:
            latex_content: String containing LaTeX code
            output_path: Path where to save the PDF (optional)
            engine: LaTeX engine to use (pdflatex, xelatex, lualatex)
            clean_aux: Remove auxiliary files after compilation
            keep_temp: Keep temporary directory for debugging

        Returns:
            Path to the generated PDF file
        """
        temp_dir = tempfile.mkdtemp(prefix="latex_")

        try:
            # Write LaTeX content to file
            tex_file = os.path.join(temp_dir, "document.tex")
            with open(tex_file, 'w', encoding='utf-8') as f:
                f.write(latex_content)

            print(f"Compiling with {engine}...")

            if self.use_persistent_container:
                # Use persistent container with docker exec
                # Clean up old files in container first
                subprocess.run(
                    ["docker", "exec", self.container_name,
                     "rm", "-f", "/tmp/document.tex", "/tmp/document.pdf", "/tmp/document.aux", "/tmp/document.log"],
                    capture_output=True  # Ignore errors if files don't exist
                )
                
                # Copy file into container
                subprocess.run(
                    ["docker", "cp", tex_file, f"{self.container_name}:/tmp/document.tex"],
                    check=True
                )

                # Run compilation inside container
                result = subprocess.run(
                    [
                        "docker", "exec", self.container_name,
                        engine,
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        "-output-directory=/tmp",
                        "/tmp/document.tex"
                    ],
                    capture_output=True,
                    text=True
                )

                # Check if PDF was created in container
                check_pdf = subprocess.run(
                    ["docker", "exec", self.container_name, "test", "-f", "/tmp/document.pdf"],
                    capture_output=True
                )
                
                if check_pdf.returncode != 0:
                    raise Exception(f"LaTeX compilation failed!\n\nOutput:\n{result.stdout}\n\nErrors:\n{result.stderr}")

                # Copy PDF back out
                pdf_temp = os.path.join(temp_dir, "document.pdf")
                subprocess.run(
                    ["docker", "cp", f"{self.container_name}:/tmp/document.pdf", pdf_temp],
                    check=True
                )

            else:
                # Use temporary container (old method)
                docker_cmd = [
                    "docker", "run",
                    "--rm",
                    "-v", f"{temp_dir}:/data",
                    "-w", "/data",
                    self.docker_image,
                    engine,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "document.tex"
                ]

                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True
                )

                pdf_temp = os.path.join(temp_dir, "document.pdf")

            # Check if PDF was generated
            if not os.path.exists(pdf_temp):
                raise Exception(f"Compilation failed!\n\nStdout:\n{result.stdout}\n\nStderr:\n{result.stderr}")

            # Copy PDF to final location
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(pdf_temp, output_path)
                final_path = output_path
            else:
                final_path = Path(f"output_{os.getpid()}.pdf")
                shutil.copy(pdf_temp, final_path)

            print(f"PDF generated successfully: {final_path}")
            return str(final_path)

        except Exception as e:
            raise Exception(f"Error during compilation: {str(e)}")

        finally:
            if not keep_temp:
                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                print(f"Temporary files kept at: {temp_dir}")

    def compile_latex_file(self, tex_file_path, output_path=None, engine="pdflatex"):
        """
        Compile a LaTeX file to PDF.

        Args:
            tex_file_path: Path to the .tex file
            output_path: Path where to save the PDF (optional)
            engine: LaTeX engine to use

        Returns:
            Path to the generated PDF file
        """
        with open(tex_file_path, 'r', encoding='utf-8') as f:
            latex_content = f.read()

        if output_path is None:
            output_path = Path(tex_file_path).with_suffix('.pdf')

        return self.compile_latex(latex_content, output_path, engine)

    def stop_container(self):
        """Stop the persistent container (call this when done)."""
        if self.use_persistent_container:
            try:
                subprocess.run(
                    ["docker", "stop", self.container_name],
                    check=True
                )
                print(f"Container {self.container_name} stopped.")
            except subprocess.CalledProcessError:
                pass

    def remove_container(self):
        """Remove the persistent container completely."""
        if self.use_persistent_container:
            try:
                subprocess.run(
                    ["docker", "rm", "-f", self.container_name],
                    check=True
                )
                print(f"Container {self.container_name} removed.")
            except subprocess.CalledProcessError:
                pass


# Example usage
if __name__ == "__main__":
    # Initialize with persistent container (faster for repeated use)
    converter = LaTeXToPDF(use_persistent_container=True)

    # Compile multiple documents quickly
    latex_code = r"""
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
% \usepackage{arxiv} % This style file can cause errors if not installed locally.

\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{url}
\usepackage{hyperref}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows}

\title{\textcolor{red}{YerleşAI: An AI-Powered System for CREATINGGG Personalized Domestic Relocation Reports in Turkey}}

\author{
  Your Name Here \\
  \textit{Your Department or Affiliation} \\
  \textit{Your University or Company}\\
  City, Turkey \\
  \texttt{your.email@example.com} \\
}

\date{October 10, 2025}
% The following lines are removed as they depend on the arxiv.sty package.
% \renewcommand{\undertitle}{}
% \renewcommand{\headerstyle}{\hrule\vspace{1em}}

\begin{document}
\maketitle

\begin{abstract}
Domestic relocation within Turkey presents significant challenges for individuals and families, including information overload, difficulty in comparing diverse regions, and a lack of personalized decision-making tools. This paper introduces "YerleşAI," a novel AI-powered system designed to mitigate these challenges by generating comprehensive, data-driven, and personalized relocation reports. The system architecture integrates a multi-source data aggregation module, a user-profiling engine based on lifestyle and economic preferences, and an AI core that leverages machine learning for predictive analytics and natural language generation (NLG) for report creation. By synthesizing data from real estate markets, cost of living indices, educational institution ratings, healthcare facility availability, and local demographic statistics, YerleşAI provides users with a ranked list of suitable destinations and a detailed narrative analysis for each. This paper outlines the system's design, methodology, a practical use-case scenario, and discusses its potential to revolutionize the decision-making process for domestic movers in Turkey.
\end{abstract}

\paragraph{Keywords:} Artificial Intelligence, Relocation Assistance, Decision Support System, Natural Language Generation, Machine Learning, Data Aggregation, Turkey


\section{Introduction}
Internal migration in Turkey is a persistent demographic trend, driven by economic opportunities, educational pursuits, lifestyle changes, and social factors \cite{tuik2023}. Annually, millions of citizens relocate between cities and regions, facing a complex and stressful decision-making process. Prospective movers must gather and analyze vast amounts of disparate information concerning housing, employment, education, healthcare, transportation, and quality of life.

Currently, this information is fragmented across various platforms: real estate websites, government portals, social media groups, and news articles. This fragmentation makes direct, objective comparison between potential destinations difficult and time-consuming. Furthermore, existing tools lack the capability to provide recommendations tailored to an individual's or a family's unique financial situation, priorities, and lifestyle preferences.

To address this gap, we propose YerleşAI, an intelligent domestic relocation assistant for Turkey. YerleşAI is an AI-driven platform that automates the process of data collection, analysis, and synthesis. It generates a personalized, easy-to-understand report that empowers users to make informed decisions. This paper details the architecture of the YerleşAI system, the underlying AI methodologies, and its expected impact on simplifying the domestic relocation journey.

\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=2.5cm]
    \tikzstyle{startstop} = [rectangle, rounded corners, minimum width=3cm, minimum height=1cm,text centered, draw=black]
    \tikzstyle{arrow} = [thick,->,>=stealth]

    \node (agg) [startstop] {Data Aggregation};
    \node (prof) [startstop, right of=agg, xshift=2cm] {User Profiling Engine};
    \node (core) [startstop, below of=agg, xshift=2.5cm, yshift=-1cm] {AI Core};
    \node (gen) [startstop, below of=core, yshift=-1cm] {Report Generation};

    \draw [arrow] (agg) -- (core);
    \draw [arrow] (prof) -- (core);
    \draw [arrow] (core) -- (gen);
\end{tikzpicture}
\caption{System Architecture of the YerleşAI Platform.}
\label{fig_sim}
\end{figure}

\section{Related Work}
The domain of relocation assistance has traditionally been dominated by manual research and real estate agencies. In the digital realm, several platforms offer partial solutions. Websites like Sahibinden and Hepsiemlak provide extensive real estate listings but lack comprehensive neighborhood analytics or cost-of-living comparisons \cite{sahibinden}. Government portals such as the Turkish Statistical Institute (TÜİK) offer valuable demographic and economic data, but it is often presented in raw formats that are not easily digestible for the average user.

While general-purpose AI assistants and large language models (LLMs) can answer specific queries about different cities, they do not offer a structured, multi-faceted comparison based on a user's complete profile. The novelty of YerleşAI lies in its holistic approach: it is a specialized decision support system that integrates diverse datasets and uses a combination of machine learning and natural language generation (NLG) to produce a bespoke analytical product, a feature not present in existing tools \cite{gatt2018}.

\section{System Architecture and Methodology}
The YerleşAI system is designed with a modular architecture, as shown in Fig. \ref{fig_sim}. It consists of four main components: the Data Aggregation Module, the User Profiling Engine, the AI Core, and the Report Generation Module.

\subsection{Data Aggregation Module}
This module is responsible for collecting, cleaning, and structuring data from a wide array of public and private sources. Web scraping, API integrations, and partnerships are utilized to gather near-real-time information. Key data categories include:
\begin{itemize}
    \item \textbf{Real Estate Data:} Property sale and rental prices, market trends, and housing type availability.
    \item \textbf{Economic Indicators:} City- and district-level cost of living, average salaries by sector, and unemployment rates.
    \item \textbf{Education and Healthcare:} School ratings, university program details, locations and specializations of hospitals and clinics.
    \item \textbf{Lifestyle and Amenities:} Crime rates, public transportation networks, availability of parks, cultural venues, and restaurants.
    \item \textbf{Demographic Data:} Population density, age distribution, and family composition from sources like TÜİK.
\end{itemize}

\subsection{User Profiling Engine}
The system's personalization capabilities stem from this engine. Users interact with a comprehensive web-based questionnaire to create a detailed profile. Input parameters include:
\begin{itemize}
    \item \textbf{Financial Profile:} Monthly budget for housing, income level, and savings.
    \item \textbf{Household Composition:} Marital status, number and age of children.
    \item \textbf{Career and Education:} Profession, industry, and educational requirements for children.
    \item \textbf{Lifestyle Preferences:} A weighted scale for priorities such as "vibrant city life" vs. "quiet and suburban," "proximity to nature," "short commute," and cultural interests.
\end{itemize}

\subsection{AI Core}
The AI Core is the brain of YerleşAI, processing the aggregated data in the context of the user's profile. It employs several machine learning models:
\begin{enumerate}
    \item \textbf{Recommender System:} A hybrid filtering model that matches the user's profile to a database of Turkish districts and cities. It calculates a "suitability score" for each potential location based on the user's weighted preferences.
    \item \textbf{Predictive Analytics:} Time-series forecasting models (e.g., ARIMA) are used to project future trends in local real estate markets, providing users with investment insights \cite{box2015}.
    \item \textbf{Natural Language Generation (NLG):} Once the top locations are ranked, an NLG model synthesizes the structured data (scores, prices, statistics) into a coherent, human-readable narrative. This model is trained to produce objective, descriptive text, such as "With a budget of 25,000 TL for rent, the Karşıyaka district in İzmir offers a 28\% higher probability of finding a 3-bedroom apartment compared to the Kadıköy district in İstanbul."
\end{enumerate}

\subsection{Report Generation Module}
This final module assembles the outputs from the AI Core into a downloadable PDF report. The report is structured to be clear and actionable, featuring:
\begin{itemize}
    \item An executive summary with the top 3 recommended locations.
    \item A detailed breakdown of each recommendation, covering housing, cost of living, pros and cons, and a final suitability score.
    \item Data visualizations, including maps of amenities, cost comparison charts, and housing market trend graphs.
\end{itemize}

\section{Use Case Scenario}
Consider a family of four with two school-aged children, moving from Ankara. The primary earner is a software developer, and their housing budget is 30,000 TL/month for rent. Their key priorities are: 1) high-quality public schools, 2) a commute time of under 45 minutes to a tech hub, and 3) access to green spaces.

The user inputs this information into the YerleşAI platform. The system processes the data and generates a report. The top recommendation might be the Bornova district in İzmir. The report would include:
\begin{itemize}
    \item A narrative summary explaining why Bornova is a strong match.
    \item A list of top-rated schools in the area with their proximity.
    \item An analysis of the average rent for a 3-bedroom apartment, noting that it falls within their budget.
    \item A map showing tech companies in and around Bornova, with estimated commute times.
    \item A section on lifestyle, highlighting local parks and cultural centers.
\end{itemize}
This provides the family with a holistic and actionable basis for their decision, far beyond what manual research could yield in the same amount of time.

\section{Challenges and Future Work}
The development of YerleşAI is not without challenges. Maintaining the accuracy and timeliness of data from myriad sources is a significant technical hurdle. Ensuring data privacy and avoiding algorithmic bias (e.g., unfairly down-ranking certain neighborhoods) are critical ethical considerations that must be addressed through transparent modeling and regular audits.

Future work will focus on several key areas. First, we plan to integrate a logistics module to provide cost estimates for moving companies. Second, a mobile application will be developed for on-the-go access. Finally, we aim to expand the model to include international relocation, assisting expatriates moving to Turkey.

\section{Conclusion}
YerleşAI represents a significant step forward in applying artificial intelligence to solve the complex real-world problem of domestic relocation. By integrating comprehensive data aggregation with advanced personalization and report generation capabilities, the system offers a powerful tool for individuals and families navigating this challenging life event in Turkey. The proposed platform has the potential to reduce stress, save time, and empower users to make optimal decisions based on data-driven insights rather than conjecture.

\bibliographystyle{unsrt}
\bibliography{references}

\end{document}
"""

    try:
        # First compilation
        pdf_path = converter.compile_latex(latex_code, output_path="../output1.pdf")
        print(f"✓ First PDF: {pdf_path}")

        # Second compilation (much faster!)
        # pdf_path = converter.compile_latex(latex_code, output_path="output2.pdf")
        # print(f"✓ Second PDF: {pdf_path}")

    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        # Optional: Stop container when done
        # converter.stop_container()
        # Or remove it completely:
        # converter.remove_container()
        pass