"use client";

import { useEffect, useState } from "react";
import type { ChatResponse, HistoryItem } from "@/types";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export default function ChatPage() {
  const [prompt, setPrompt] = useState("");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [selected, setSelected] = useState<HistoryItem | null>(null);
  const [modalItem, setModalItem] = useState<HistoryItem | null>(null);
  const [loading, setLoading] = useState(false);

  // Load history from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem("fantasy_history");
    if (stored) {
      try {
        const items: HistoryItem[] = JSON.parse(stored);
        setHistory(items);
        if (items.length) setSelected(items[items.length - 1]);
      } catch {
        /* ignore */
      }
    }
  }, []);

  // Persist history when it changes
  useEffect(() => {
    localStorage.setItem("fantasy_history", JSON.stringify(history));
  }, [history]);

  const sendPrompt = async () => {
    if (!prompt.trim()) return;
    try {
      setLoading(true);
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data: ChatResponse = await res.json();
      const item: HistoryItem = { ...data, prompt };
      setHistory((prev) => [...prev, item]);
      setSelected(item);
      setPrompt("");
    } catch (err) {
      console.error(err);
      alert("Failed to get response from server");
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendPrompt();
    }
  };

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <aside className="w-48 sm:w-56 bg-slate-900 text-white p-2 overflow-y-auto space-y-2">
        {history.map((item) => (
          <button
            key={item.id}
            onClick={() => {
              setSelected(item);
              setModalItem(item);
            }}
            className="w-full aspect-square rounded-lg overflow-hidden border-2 border-transparent hover:border-purple-400 focus:border-purple-400"
          >
            <img
              src={`${BACKEND_URL}${item.image_url}`}
              alt="thumb"
              className="object-cover w-full h-full"
            />
          </button>
        ))}
      </aside>

      {/* Main panel */}
      <main className="flex-1 flex flex-col">
        {/* Output area */}
        <div className="flex-1 flex flex-col items-center justify-center p-4 overflow-auto">
          {selected ? (
            <div className="max-w-lg w-full flex flex-col items-center space-y-4">
              <img
                src={`${BACKEND_URL}${selected.image_url}`}
                alt="fantasy"
                className="w-full rounded-lg shadow-lg"
              />
              <p className="text-lg text-center whitespace-pre-wrap">
                {selected.answer}
              </p>
            </div>
          ) : (
            <p className="text-gray-500">Ask the oracle of Altheria...</p>
          )}
        </div>

        {/* Input */}
        <div className="p-4 border-t flex gap-2">
          <input
            className="flex-1 border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
            placeholder="Speak, adventurer..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKey}
            disabled={loading}
          />
          <button
            className="bg-purple-600 text-white px-4 py-2 rounded-lg disabled:opacity-50"
            onClick={sendPrompt}
            disabled={loading}
          >
            {loading ? "..." : "Send"}
          </button>
        </div>
        {/* Modal */}
        {modalItem && (
          <div
            className="fixed inset-0 flex items-center justify-center bg-black/60 z-50"
            onClick={() => setModalItem(null)}
          >
            <div
              className="bg-white rounded-lg p-4 max-w-lg w-[90%] shadow-xl relative"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                className="absolute top-2 right-2 text-gray-500 hover:text-gray-700"
                onClick={() => setModalItem(null)}
              >
                ✕
              </button>
              <img
                src={`${BACKEND_URL}${modalItem.image_url}`}
                alt="full"
                className="w-full rounded-md mb-4"
              />
              <h2 className="font-semibold mb-2">{modalItem.prompt}</h2>
              <p className="whitespace-pre-wrap text-sm">{modalItem.answer}</p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
