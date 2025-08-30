export type NEREngine = 'auto' | 'hf' | 'local' | 'spacy' | 'hybrid'

export type Entity = { type: string; placeholder: string; value_preview: string }

export type RedactResponse = {
  redacted_text: string
  entities: Entity[]
  map: Record<string, string>
}

export type NERStatus = {
  spacy_available: boolean
  transformers_available: boolean
  hf_configured: boolean
  hf_models: string[]
  local_model_configured: boolean
  // Optional fields exposed by backend for LLM fallback visibility
  fallback_provider?: string
  fallback_model?: string
  fallback_enabled?: boolean
  // Offline/aux diagnostics
  strict_privacy?: boolean
  aux_local_models?: string[]
  spacy_model?: string
}

export interface ClientOptions {
  baseUrl?: string // e.g., 'http://localhost:8000' when running on device
  timeoutMs?: number // default 15000
  fetchImpl?: typeof fetch // allow custom fetch in Lynx
}

export class PrivacyGuardianClient {
  private baseUrl: string
  private timeoutMs: number
  private fetchImpl: typeof fetch

  constructor(opts?: ClientOptions) {
    this.baseUrl = (opts?.baseUrl || '').replace(/\/$/, '')
    this.timeoutMs = opts?.timeoutMs ?? 15000
    // Ensure fetch is correctly bound to Window in browsers to avoid Illegal invocation errors
    if (opts?.fetchImpl) {
      this.fetchImpl = opts.fetchImpl
    } else if (typeof window !== 'undefined' && typeof window.fetch === 'function') {
      this.fetchImpl = window.fetch.bind(window)
    } else {
      this.fetchImpl = fetch
    }
  }

  private withTimeout<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
    if (this.timeoutMs <= 0) return promise
    return new Promise<T>((resolve, reject) => {
      const id = setTimeout(() => reject(new Error('Request timed out')), this.timeoutMs)
      promise.then((v) => { clearTimeout(id); resolve(v) }).catch((e) => { clearTimeout(id); reject(e) })
      if (signal) signal.addEventListener('abort', () => { clearTimeout(id); reject(new Error('Aborted')) })
    })
  }

  async nerStatus(signal?: AbortSignal): Promise<NERStatus> {
    const url = `${this.baseUrl}/ner_status`
    const res = await this.withTimeout(this.fetchImpl(url, { signal }), signal)
    if (!('ok' in res) || !(res as any).ok) throw new Error(`NER status failed`)
    return (res as Response).json()
  }

  async version(signal?: AbortSignal): Promise<{ version?: string; commit?: string | null; started_at?: string }>{
    const url = `${this.baseUrl}/version`
    const res = await this.withTimeout(this.fetchImpl(url, { signal }), signal)
    if (!('ok' in res) || !(res as any).ok) throw new Error(`Version check failed`)
    return (res as Response).json()
  }

  async redact(text: string, nerEngine: NEREngine = 'auto', signal?: AbortSignal): Promise<RedactResponse> {
    const url = `${this.baseUrl}/redact`
    const res = await this.withTimeout(this.fetchImpl(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, ner_engine: nerEngine }),
      signal,
    }), signal)
    if (!('ok' in res) || !(res as any).ok) {
      const txt = await (res as Response).text().catch(() => '')
      throw new Error(txt || 'Redact failed')
    }
    return (res as Response).json()
  }
}

// Convenience singleton for simple usage
export const privacyGuardian = new PrivacyGuardianClient()


