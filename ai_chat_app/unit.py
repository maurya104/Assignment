import unittest
from unittest.mock import patch, MagicMock
from app import get_pdf_text, get_text_chunks, get_vector_store, user_input, get_conversational_chain  # Importing from app.py

class TestPDFChatbot(unittest.TestCase):

    @patch("PyPDF2.PdfReader")
    def test_get_pdf_text(self, MockPdfReader):
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock(extract_text=MagicMock(return_value="Page 1 text")),
                          MagicMock(extract_text=MagicMock(return_value="Page 2 text"))]
        MockPdfReader.return_value = mock_pdf
        
        pdf_docs = ["D:\Downloads\interview preparation\ml.pdf"]
        
        text = get_pdf_text(pdf_docs)
        
        self.assertEqual(text, "Page 1 textPage 2 text")

    def test_get_text_chunks(self):
        text = "This is a long document. It will be split into chunks."
        
        chunks = get_text_chunks(text)
        
        self.assertTrue(len(chunks) > 0)

    @patch("langchain_community.vectorstores.FAISS.from_texts")
    @patch("langchain_google_genai.GoogleGenerativeAIEmbeddings")
    def test_get_vector_store(self, MockEmbeddings, MockFAISS):
        mock_embeddings = MagicMock()
        MockEmbeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        MockFAISS.from_texts.return_value = mock_vector_store
        
        text_chunks = ["This is chunk 1", "This is chunk 2"]
        
        get_vector_store(text_chunks)
        
        MockFAISS.from_texts.assert_called_once_with(text_chunks, embedding=mock_embeddings)
        mock_vector_store.save_local.assert_called_once_with("faiss_index")

    @patch("langchain_community.vectorstores.FAISS.load_local")
    @patch("langchain_google_genai.GoogleGenerativeAIEmbeddings")
    @patch("langchain_community.vectorstores.FAISS.similarity_search")
    def test_user_input(self, MockSimilaritySearch, MockEmbeddings, MockLoadLocal):
        mock_embeddings = MagicMock()
        MockEmbeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        MockLoadLocal.return_value = mock_vector_store
        
        MockSimilaritySearch.return_value = ["document1", "document2"]
        
        user_question = "What is the document about?"
        
        user_input(user_question)

        print("MockSimilaritySearch call count:", MockSimilaritySearch.call_count)
        
        MockSimilaritySearch.assert_called_once_with(user_question)
        
        self.assertEqual(MockSimilaritySearch.call_count, 1)

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_get_conversational_chain(self, MockChat):
        mock_chain = MagicMock()
        MockChat.return_value = mock_chain
        
        chain = get_conversational_chain()
        
        self.assertIsNotNone(chain)

if __name__ == "__main__":
    unittest.main()
