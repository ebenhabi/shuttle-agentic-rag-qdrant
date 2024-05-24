/**
    File parsing and embedding into Qdrant
    Next, we will implement a File struct for CSV file parsing - it should be able to hold the file path,
    contents as well as the rows as a Vec<String> (string array, or more accurately a vector of strings).
    There's a few reasons why we store the rows as a Vec<String>:
    1 - Smaller chunks improve the retrieval accuracy, one of the biggest challenges that RAG has to deal
        with. Retrieving a wrong or otherwise inaccurate document can hamper accuracy significantly.
    2 - Improved retrieval accuracy leads to enhanced contextual relevance - which is quite important for
        complex queries that require specific question.
    3 - Processing and indexing smaller chunks
*/
use anyhow::Result;
use std::path::PathBuf;

pub struct File {
    pub path: String,
    pub contents: String,
    pub rows: Vec<String>,
}

impl File {
    pub fn new(path: PathBuf) -> Result<Self> {
        let contents = std::fs::read_to_string(&path)?;

        let path_as_str = format!("{}", path.display());

        let rows = contents
            .lines()
            .map(|x| x.to_owned())
            .collect::<Vec<String>>();

          Ok(Self {
            path: path_as_str,
            contents,
            rows
          })
    }
}