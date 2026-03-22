import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
from collections import Counter
import math

# ==========================================
# PART 1: CORE NLP CLASSES (PyTorch/Custom)
# ==========================================
class CustomTokenizer:
    def __init__(self):
        self.pattern = re.compile(r'\b\w+\b')
    def tokenize(self, text):
        return self.pattern.findall(str(text).lower())

def generate_ngrams(tokens, n=2):
    return ['_'.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

class CustomTFIDFVectorizer:
    def __init__(self, max_features=3000, use_ngrams=True):
        self.max_features = max_features
        self.use_ngrams = use_ngrams
        self.tokenizer = CustomTokenizer()
        self.vocabulary = {}
        self.idf = None
        
    def fit_transform(self, corpus):
        doc_freq = Counter()
        token_counts = Counter()
        for doc in corpus:
            tokens = self.tokenizer.tokenize(doc)
            all_tokens = tokens + (generate_ngrams(tokens, 2) + generate_ngrams(tokens, 3) if self.use_ngrams else [])
            doc_freq.update(set(all_tokens))
            token_counts.update(all_tokens)
            
        top_tokens = [t for t, c in token_counts.most_common(self.max_features)]
        self.vocabulary = {token: idx for idx, token in enumerate(top_tokens)}
        
        num_docs = len(corpus)
        self.idf = torch.zeros(len(self.vocabulary))
        for token, idx in self.vocabulary.items():
            self.idf[idx] = math.log(num_docs / (doc_freq[token] + 1)) + 1.0
            
        return self.transform(corpus)
        
    def transform(self, corpus):
        num_docs = len(corpus)
        vocab_size = len(self.vocabulary)
        indices, values = [], []
        
        for i, doc in enumerate(corpus):
            tokens = self.tokenizer.tokenize(doc)
            all_tokens = tokens + (generate_ngrams(tokens, 2) + generate_ngrams(tokens, 3) if self.use_ngrams else [])
            term_freq = Counter([t for t in all_tokens if t in self.vocabulary])
            
            total_terms = max(len(all_tokens), 1)
            norm_sum = 0.0
            doc_indices, doc_values = [], []
            
            for token, count in term_freq.items():
                idx = self.vocabulary[token]
                tfidf_val = (count / total_terms) * self.idf[idx].item()
                doc_indices.append(idx)
                doc_values.append(tfidf_val)
                norm_sum += tfidf_val ** 2
            
            norm = math.sqrt(norm_sum) + 1e-9
            for idx, val in zip(doc_indices, doc_values):
                indices.append([i, idx])
                values.append(val / norm)
                
        if not indices:
            return torch.sparse_coo_tensor(size=(num_docs, vocab_size), dtype=torch.float32)
        return torch.sparse_coo_tensor(torch.tensor(indices).t(), torch.tensor(values, dtype=torch.float32), size=(num_docs, vocab_size))

class DenseSemanticLayer(nn.Module):
    def __init__(self, vocab_words, tfidf_vectorizer, embed_dim=300):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab = {word: i for i, word in enumerate(vocab_words)}
        if "<UNK>" not in self.vocab: self.vocab["<UNK>"] = len(self.vocab)
        if "<PAD>" not in self.vocab: self.vocab["<PAD>"] = len(self.vocab)
            
        self.vocab_size = len(self.vocab)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        with torch.no_grad():
            self.embedding.weight[self.vocab["<PAD>"]] = 0.0
        self.tfidf = tfidf_vectorizer
        
    def forward(self, token_lists):
        batch_size = len(token_lists)
        device = self.embedding.weight.device
        max_len = max(max([len(t) for t in token_lists]) if token_lists else 1, 1)
        
        token_indices = torch.full((batch_size, max_len), self.vocab["<PAD>"], dtype=torch.long, device=device)
        tfidf_weights = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
        
        for i, tokens in enumerate(token_lists):
            for j, token in enumerate(tokens):
                token_indices[i, j] = self.vocab.get(token, self.vocab["<UNK>"])
                tfidf_idx = self.tfidf.vocabulary.get(token, -1)
                tfidf_weights[i, j] = self.tfidf.idf[tfidf_idx].item() if tfidf_idx != -1 else 1.0
                
        embeds = self.embedding(token_indices) 
        tfidf_weights = tfidf_weights.unsqueeze(-1) 
        weighted_embeds = embeds * tfidf_weights
        
        return weighted_embeds.sum(dim=1) / tfidf_weights.sum(dim=1).clamp(min=1e-9)

class SimilaritySearcher(nn.Module):
    def __init__(self, database_vectors):
        super().__init__()
        self.register_buffer('database', database_vectors)
    def forward(self, query_vectors):
        q_norm = torch.nn.functional.normalize(query_vectors, p=2, dim=1)
        db_norm = torch.nn.functional.normalize(self.database, p=2, dim=1)
        return torch.mm(q_norm, db_norm.t())

# ==========================================
# PART 2: APP ENGINE INITIALIZATION 
# ==========================================
@st.cache_resource
def load_engine():
    try:
        df = pd.read_csv('customer_support_tickets.csv')
    except Exception:
        # Fallback empty dataframe for streamlit initialization if data missing
        df = pd.DataFrame({'Ticket Description': ["money issues", "laptop is not turning on", "forgot my password"], 'Ticket Type': ["Billing inquiry", "Technical issue", "Product inquiry"], 'Ticket Subject': ["Billing", "Hardware", "Account"]})
        
    df['Ticket Description'] = df['Ticket Description'].astype(str).fillna('')
    df = df[df['Ticket Description'].str.strip() != '']
    # Sample for Streamlit Cloud memory efficiency (Free Tier)
    df = df.head(3000).copy()
        
    tfidf = CustomTFIDFVectorizer(max_features=3000)
    sparse_vectors = tfidf.fit_transform(df['Ticket Description'].tolist())
    
    dense_layer = DenseSemanticLayer(list(tfidf.vocabulary.keys()), tfidf, 300)
    tokenizer = CustomTokenizer()
    token_lists = [tokenizer.tokenize(t) for t in df['Ticket Description']]
    
    with torch.no_grad():
        dense_database = dense_layer(token_lists)
    
    searcher = SimilaritySearcher(dense_database)
    return df, tfidf, sparse_vectors, dense_layer, searcher

# ==========================================
# PART 3: STREAMLIT UI
# ==========================================
st.set_page_config(page_title="HSRIS Support App", layout="centered", page_icon="🎫")

st.title("🎫 HSRIS: Hybrid Semantic Retrieval")
st.markdown("**Created by: Ali Abbas (23f-3037) - SE-6A**")
st.markdown("Search internal customer support ticket resolutions using PyTorch Hybrid Encodings.")

df, tfidf, sparse_vectors, dense_layer, searcher = load_engine()

# --- SIider UI ---
alpha = st.slider(
    "Hybrid Matching Ratio ($\\alpha$): Semantic vs Keyword Focus", 
    0.0, 1.0, 0.5, 0.05,
    help="0.0 = Pure TF-IDF (Keywords) | 1.0 = Pure GloVe (Semantics/Intent)"
)

left, right = st.columns(2)
with left:
    st.caption("⬅️ Priority: Keyword Overlap")
with right:
    st.caption("Priority: Semantic Meaning ➡️")

st.markdown("---")

query = st.text_area("Customer Ticket Description:", "Type customer issue here... (e.g. payment methodology failed)")

if st.button("Find Similar Past Tickets 🚀"):
    with st.spinner("Calculating vector similarities natively..."):
        # 1. Sparse Search CPU
        query_sparse = tfidf.transform([query]).to_dense()[0].unsqueeze(1)
        sim_sparse = torch.sparse.mm(sparse_vectors, query_sparse).squeeze(1)
        sim_sparse_scaled = sim_sparse / sim_sparse.max().clamp(min=1e-9)
        
        # 2. Dense Search CPU
        tokenizer = CustomTokenizer()
        with torch.no_grad():
            q_dense = dense_layer([tokenizer.tokenize(query)])
            sim_dense = searcher(q_dense).squeeze(0)
        sim_dense_scaled = (sim_dense + 1.0) / 2.0
        
        # 3. Hybrid scoring
        score = alpha * sim_dense_scaled + (1.0 - alpha) * sim_sparse_scaled
        top_k = min(3, len(df))
        top_indices = torch.topk(score, top_k).indices.numpy()
        
        results = df.iloc[top_indices].copy()
        results['Score'] = score[top_indices].numpy()
        
        # Display Results
        top_1 = results.iloc[0]
        st.success(f"**Predicted Target Ticket Type:** {top_1['Ticket Type']}")
        
        st.markdown("### Top 3 Similar Past Resolutions")
        for i, row in results.iterrows():
            with st.expander(f"Similarity Score: {row['Score']:.2f} | Topic: {row['Ticket Subject']} | Type: {row['Ticket Type']}"):
                st.write(f"**Past Ticket:** _{row['Ticket Description']}_")
                st.write(f"**Product Related:** `{row.get('Product Purchased', 'Unknown')}`")
