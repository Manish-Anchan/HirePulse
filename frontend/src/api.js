export async function fetchThreads() {
  const res = await fetch("/api/threads");
  if (!res.ok) throw new Error(`Failed to load threads (${res.status})`);
  const data = await res.json();
  return data.threads || [];
}

export async function fetchThreadMessages(threadId) {
  const res = await fetch(`/api/threads/${encodeURIComponent(threadId)}/messages`);
  if (!res.ok) throw new Error(`Failed to load messages (${res.status})`);
  const data = await res.json();
  return {
    messages: data.messages || [],
    files: data.files || [],
  };
}

export async function deleteThread(threadId) {
  const res = await fetch(`/api/threads/${encodeURIComponent(threadId)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`Failed to delete thread (${res.status})`);
  return res.json();
}

export async function uploadDocuments(files, threadId = null) {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  if (threadId) {
    formData.append("thread_id", threadId);
  }

  const res = await fetch("/api/documents/upload", {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || `Upload failed (${res.status})`);
  }
  return data;
}

export async function streamChat({
  threadId,
  message,
  onMeta,
  onToken,
  onDone,
  onError,
  signal,
}) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, message }),
    signal,
  });

  if (!res.ok) {
    throw new Error(`Chat request failed (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const rawEvent of events) {
      const lines = rawEvent.split("\n");
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (!payload) continue;

        let evt;
        try {
          evt = JSON.parse(payload);
        } catch {
          continue;
        }

        if (evt.type === "meta" && onMeta) onMeta(evt);
        else if (evt.type === "token" && onToken) onToken(evt.content);
        else if (evt.type === "done" && onDone) onDone(evt);
        else if (evt.type === "error" && onError) onError(evt.error);
      }
    }
  }
}
