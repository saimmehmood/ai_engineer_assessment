import { useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

function cleanMarkdownResponse(response) {
  let clean = response;
  if (typeof clean === 'string' && clean.startsWith('"') && clean.endsWith('"')) {
    clean = clean.slice(1, -1);
  }
  // Replace both \n and literal \n
  clean = clean.replace(/\\n/g, '\n'); // replaces \n (escaped)
  clean = clean.replace(/\n/g, '\n');   // ensures all newlines are real
  return clean;
}

function App() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const responseRef = useRef(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResponse('');
    setLoading(true);
    await handleRegularQuery();
    setLoading(false);
  };

  const handleRegularQuery = async () => {
    try {
      const res = await fetch('/v1/threads/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [
            { role: 'user', content: query }
          ]
        })
      });
      const data = await res.json();
      setResponse(
        data?.payload?.map((msg) => msg.content).join('\n\n') || 'No response.'
      );
    } catch (err) {
      setResponse('Error: ' + err.message);
    }
  };

  const cleanedResponse = cleanMarkdownResponse(response);

  return (
    <div className="container">
      <h1>Text-to-SQL Chat Demo</h1>
      <form onSubmit={handleSubmit} className="query-form">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about your financial data..."
          rows={3}
          required
        />
        <div className="controls">
          <button type="submit" disabled={loading}>
            {loading ? 'Loading...' : 'Send'}
          </button>
        </div>
      </form>
      <div className="response-area" ref={responseRef}>
        <h2>Response</h2>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{cleanedResponse}</ReactMarkdown>
      </div>
    </div>
  );
}

export default App;
