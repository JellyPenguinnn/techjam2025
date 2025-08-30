import React, { useEffect, useMemo, useState } from 'react'
import { PrivacyGuardianClient, type NERStatus, type NEREngine } from './sdk/privacyGuardian'

type Entity = { type: string; placeholder: string; value_preview: string; confidence?: number }

type RedactResponse = {
  redacted_text: string
  entities: Entity[]
  map: Record<string, string>
}

export default function App() {
  const [text, setText] = useState('')
  const [nerEngine] = useState<NEREngine>('local') // Always use local for offline-only build

  const [redacted, setRedacted] = useState<RedactResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copyMsg, setCopyMsg] = useState<string>('')
  const [clearMsg, setClearMsg] = useState<string>('')
  const [status, setStatus] = useState<NERStatus | null>(null)
  const [backendVersion, setBackendVersion] = useState<{ version?: string; commit?: string | null; started_at?: string } | null>(null)
  const [lastSeenVersion, setLastSeenVersion] = useState<string>('')

  const apiBase = (import.meta as any).env?.VITE_API_BASE || ''
  const client = useMemo(() => new PrivacyGuardianClient({ baseUrl: apiBase }), [apiBase])

  const doRedact = async () => {
    setError(null)
    setLoading(true)
    try {
      const data = await client.redact(text, nerEngine)
      setRedacted(data)
    } catch (e: any) {
      setError(e.message || 'Failed to redact')
    } finally {
      setLoading(false)
    }
  }

  const copyRedacted = async () => {
    if (!redacted?.redacted_text) return
    try {
      await navigator.clipboard.writeText(redacted.redacted_text)
      setCopyMsg('Copied!')
      setTimeout(() => setCopyMsg(''), 1500)
    } catch (e) {
      setCopyMsg('Copy failed')
      setTimeout(() => setCopyMsg(''), 1500)
    }
  }

  const clearPrompt = () => {
    const ok = window.confirm('Clear the input? This cannot be undone.')
    if (!ok) return
    setText('')
    setRedacted(null)
    setClearMsg('Cleared')
    setTimeout(() => setClearMsg(''), 1200)
  }

  // Removed mode switching - always offline

  // Status fetch on mount
  useEffect(() => {
    const ac = new AbortController()
    ;(async () => {
      try {
        const s = await client.nerStatus(ac.signal)
        setStatus(s)
        const v = await client.version(ac.signal).catch(() => null)
        if (v) {
          setBackendVersion(v)
          const fingerprint = `${v.version || ''}-${v.commit || ''}-${v.started_at || ''}`
          if (!lastSeenVersion) setLastSeenVersion(fingerprint)
        }
      } catch (_) {
        setStatus(null)
      }
    })()
    return () => ac.abort()
  }, [])

  // Removed entity counts display for cleaner UI



  const getCharCountClass = (count: number) => {
    if (count > 10000) return 'char-count over-limit'
    if (count > 9000) return 'char-count near-limit'
    return 'char-count'
  }

  // Removed auto-resize - back to manual resize

  return (
    <div className="container">
      <div className="hero">
        <div>
          <h1 className="title">Privacy Guardian</h1>
          <p className="subtitle">AI-Powered PII Detection & Redaction</p>
        </div>
      </div>

      <div className="card">
        <div className="section">
          <div className="row" style={{ justifyContent: 'flex-end' }}>
            <button className="btn ghost" onClick={clearPrompt} disabled={loading || !text}>
              Clear
            </button>
            <button className="btn primary" onClick={doRedact} disabled={loading || !text}>
              {loading ? 'Redacting…' : 'Redact'}
            </button>
          </div>
        </div>
        <div className="section">
          <textarea
            className="input"
            value={text}
            onChange={e => setText(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                e.preventDefault()
                if (!loading && text.trim() && text.length <= 10000) doRedact()
              }
            }}
            placeholder={'Paste or type text containing sensitive information...\n\nExample:\n"Contact me at john.doe@company.com or call +1-555-123-4567. My account ID is ACC-789123 and I live at 123 Main Street, Boston, MA 02101."\n\nPress Cmd/Ctrl + Enter to redact (Max 10,000 characters)'}
            aria-label="Text input for PII detection"
          />
          <div className="row" style={{ justifyContent: 'flex-end', marginTop: 12, alignItems: 'center' }}>
            <span className={getCharCountClass(text.length)}>
              {text.length}/10,000 characters
            </span>
          </div>
          {error && (
            <div style={{ 
              marginTop: '12px', 
              padding: '12px 16px', 
              background: 'var(--danger)', 
              color: 'white', 
              borderRadius: '8px',
              fontSize: '14px'
            }}>
              {error}
            </div>
          )}
          {clearMsg && (
            <div style={{ 
              marginTop: '8px', 
              color: 'var(--success)', 
              fontSize: '14px',
              fontWeight: '500'
            }}>
              {clearMsg}
            </div>
          )}
        </div>
        <div className="section grid">
          <div className="panel">
            <div className="panel-header">
              <h3>Protected Text</h3>
              <div className="row" style={{ gap: 8, alignItems: 'center' }}>
                {copyMsg && (
                  <div className="badge success" aria-live="polite">
                    {copyMsg}
                  </div>
                )}
                <button 
                  className="btn ghost" 
                  onClick={copyRedacted} 
                  disabled={!redacted?.redacted_text}
                >
                  Copy
                </button>
              </div>
            </div>
            <div className="box fill scroll">
              <pre 
                className="input autoh" 
                style={{ 
                  whiteSpace: 'pre-wrap', 
                  wordBreak: 'break-word', 
                  margin: 0,
                  background: 'transparent',
                  border: 'none',
                  padding: 0,
                  color: redacted?.redacted_text ? 'var(--text)' : 'var(--text-muted)'
                }}
              >
                {redacted?.redacted_text || 'Redacted text will appear here...'}
              </pre>
            </div>
          </div>
          
          <div className="panel">
            <div className="panel-header">
              <h3>Detected Entities</h3>
              {redacted?.entities?.length ? (
                <div className="badge">
                  {redacted.entities.length} entities found
                </div>
              ) : null}
            </div>
            <div className="box fill scroll">
              {redacted?.entities?.length ? (
                <table className="entities">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Type</th>
                      <th>Original Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {redacted.entities.map((e, i) => {
                      const full = (redacted?.map?.[e.placeholder] || e.value_preview || '') as string
                      const shown = full.length > 32 ? full.slice(0, 32) + '…' : full
                      return (
                        <tr key={i}>
                          <td>{i + 1}</td>
                          <td>
                            <span className="entity-type">
                              {e.type.replace(/_/g, ' ')}
                            </span>
                          </td>
                          <td style={{ fontFamily: 'monospace', fontSize: '13px' }}>
                            {shown}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              ) : (
                <div style={{ 
                  textAlign: 'center', 
                  color: 'var(--text-muted)', 
                  padding: '40px 20px',
                  fontSize: '14px'
                }}>
                  No sensitive data detected yet.<br />
                  Enter some text with PII to see results.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
