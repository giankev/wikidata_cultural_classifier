# wikidata_cultural_classifier

### Setting up the virtual environment and running:

**Python 3.12 has dependencies issues (related to python3-distutils), use an older version!** 

Create a python virtual environment (use the available version instead of 3.11):

> python3.11 -m venv "cultural_classifier"

Enable the environment:

> source cultural_classifier/bin/activate

Clone this repository wherever you wish:

> git clone https://github.com/giankev/wikidata_cultural_classifier.git

Change your current directory to the cultural classifier project folder:

> cd wikidata_cultural_classifier

Install the required packages:

> pip install -r requirements.txt

Run the scripts from within the python virtual environment:

> python3 script_name.py

### Pipeline

1. Dataset Modeling

2. Graèh construction

3. Graph Embedding with Graph2Vec

4. Classification Model

5. Inference on New Data

### Visualizing the generated graphs

By running the text2graph script, a web version of the graph will be generated inside the web_graphs folder with the name **title_language_graph.html**.
To change the selected wikidata item edit the following line inside text2graph main function:
> wikidata_url = "https://www.wikidata.org/wiki/{your desired wikidata id here}"
To change the language change the following line, also inside text2graph main function:
> language = "en"

### Issues

- Graphs take too long to be preprocessed