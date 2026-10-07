/**
 * fetch that fails loudly: a 404 or a dropped connection becomes an error naming
 * the file, instead of a confusing parse failure further down. `onBytes` reports
 * download progress as chunks arrive.
 */

export async function fetchOk(url: string, onBytes?: (n: number) => void): Promise<Response> {
  const file = url.split("/").pop();
  let r: Response;
  try {
    r = await fetch(url);
  } catch {
    throw new Error(`couldn't download ${file} (network error)`);
  }
  if (!r.ok) throw new Error(`couldn't download ${file} (HTTP ${r.status})`);
  if (!onBytes || !r.body) return r;
  const chunks: Uint8Array[] = [];
  const reader = r.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    onBytes(value.byteLength);
  }
  return new Response(new Blob(chunks), { headers: r.headers });
}
