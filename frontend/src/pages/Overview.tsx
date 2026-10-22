import { useState } from "react";
import { Link } from "react-router-dom";
import { gql, Q } from "../api";
import { useAsync } from "../lib/useAsync";

interface Arch {
  layers: Array<{ id: string; label: string; items: string[] }>;
  modules: Array<{ id: string; label: string; path: string; layers: string[] }>;
}
const MODULE_LINK: Record<string, string> = { change: "/change", pdm: "/pdm", safety: "/safety", quote: "/quote", mro: "/mro", ap: "/ap" };

export function OverviewPage() {
  const arch = useAsync(() => gql<{ architecture: Arch }>(Q.architecture));
  const rep = useAsync(() => gql<{ evalReport: any }>(Q.report).then((d) => d.evalReport)); // eslint-disable-line @typescript-eslint/no-explicit-any
  const [hot, setHot] = useState<string>("change");
  const a = arch.data?.architecture;
  const m = a?.modules.find((x) => x.id === hot);
  const lit = new Set(m?.layers ?? []);
  const r = rep.data;
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Aerospace engineering decisions you can trace back to evidence</h1>
          <p className="lede">AeroNexus reads drawing revisions, engine sensor histories, maintenance logbooks, safety reports and
            supplier invoices. It checks them with deterministic rules, predicts with models trained on public aviation data, and
            drafts ERP, CRM and maintenance actions. Anything uncertain is marked for a person, and nothing is released without a
            named approver.</p>
        </div>
      </div>
      {arch.error && <div className="err">{arch.error}</div>}
      {r && (
        <div className="figures">
          {r.pdm?.available && <div className="figure"><b>{r.pdm.fd001_rmse}</b><span>cycles RUL error on NASA C-MAPSS FD001</span></div>}
          {r.safety?.available && <div className="figure"><b>{r.safety.test_auroc}</b><span>safety triage AUROC on unseen NTSB years</span></div>}
          <div className="figure"><b>{Math.round(r.change.receipt_reproducibility * 100)}%</b><span>change-order numbers that re-compute exactly</span></div>
          <div className="figure"><b>{r.ap.false_auto_match}</b><span>bad invoices auto-matched</span></div>
          <div className={`figure ${r.gate.passed ? "" : "alert"}`}><b>{r.gate.checks.filter((c: { passed: boolean }) => c.passed).length}/{r.gate.checks.length}</b><span>release gates passing</span></div>
        </div>
      )}
      {a && (
        <section className="section">
          <h2>How it works</h2>
          <p className="small muted">Pick a workflow to see which stages it uses.</p>
          <div className="row" style={{ marginBottom: 14 }}>
            {a.modules.map((x) => (
              <button key={x.id} className={`btn ${hot === x.id ? "primary" : ""}`} onClick={() => setHot(x.id)}>{x.label}</button>
            ))}
          </div>
          <div className="panel flush">
            <div className="arch" style={{ gridTemplateColumns: `repeat(${a.layers.length}, minmax(118px, 1fr))` }}>
              {a.layers.map((l, i) => (
                <div key={l.id} className={`layer ${lit.has(l.id) ? "hot" : ""}`}>
                  <div className="step">Stage {i + 1}</div>
                  <h3>{l.label}</h3>
                  <ul>{l.items.map((it) => <li key={it}>{it}</li>)}</ul>
                </div>
              ))}
            </div>
          </div>
          <p className="loopnote">{m?.path}. Reviewer corrections become new labelled data, and the assurance stage re-tests every
            stage on each release. <Link to={MODULE_LINK[hot]}>Open this workflow</Link></p>
        </section>
      )}
      <section className="section two">
        <div className="panel">
          <h3>Reliable by construction</h3>
          <ul className="small" style={{ paddingLeft: 18, margin: 0 }}>
            <li>Numbers come from versioned formulas with receipts, never from a language model</li>
            <li>Model answers must appear word for word in the document or they are discarded</li>
            <li>Every case is an event log: it can be replayed and audited, and failed steps resume where they stopped</li>
            <li>Retrieval only returns the policy revision in force on the date, and flags contradictions</li>
          </ul>
        </div>
        <div className="panel">
          <h3>Responsible by default</h3>
          <ul className="small" style={{ paddingLeft: 18, margin: 0 }}>
            <li>Roles decide who can run, approve and evaluate; agents can never approve</li>
            <li>Safety and maintenance outputs are decision support; qualified people decide</li>
            <li>Every dataset records its source, license and checksum; synthetic results are labelled as such</li>
            <li>Customer messages say an AI assistant prepared them</li>
          </ul>
        </div>
      </section>
    </>
  );
}
