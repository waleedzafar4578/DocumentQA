from langchain_core.documents import Document
from sqlalchemy.testing.suite.test_reflection import metadata
# this git repo which follow for build pipeline
# https://github.com/krishnaik06/RAG-Tutorials/blob/main/src/data_loader.py
doc =Document(
    page_content="i'll show u later",
    metadata={
        "Source":"pata nhi"
    }
)

def main():
    print(doc)





if __name__ == '__main__':
    main()

