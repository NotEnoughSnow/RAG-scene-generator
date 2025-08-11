"use client";
import { useEffect, useState } from "react";

export default function Home() {
  const [message, setMessage] = useState("Loading...");

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/hello")
      .then(res => res.json())
      .then(data => setMessage(data.message))
      .catch(err => setMessage("Error: " + err.message));
  }, []);

  return (
    <main style={{ padding: "2rem" }}>
      <h1>{message}</h1>
      <p>This is the Next.js frontend talking to the FastAPI backend.</p>
    </main>
  );
}