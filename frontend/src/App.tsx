import { lazy, Suspense, useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { gql, identity, Q, setIdentity, type Approval, type Identity, type Role } from "./api";
import { useAsync } from "./lib/useAsync";

const ChatPanel = lazy(() => import("./components/ChatPanel").then((m) => ({ default: m.ChatPanel })));
const OverviewPage = lazy(() => import("./pages/Overview").then((m) => ({ default: m.OverviewPage })));
const ChangeOrdersPage = lazy(() => import("./pages/ChangeOrders").then((m) => ({ default: m.ChangeOrdersPage })));
const PdmPage = lazy(() => import("./pages/Pdm").then((m) => ({ default: m.PdmPage })));
const SafetyPage = lazy(() => import("./pages/Safety").then((m) => ({ default: m.SafetyPage })));
const QuotePage = lazy(() => import("./pages/Quote").then((m) => ({ default: m.QuotePage })));
const MroPage = lazy(() => import("./pages/Mro").then((m) => ({ default: m.MroPage })));
const ApPage = lazy(() => import("./pages/Ap").then((m) => ({ default: m.ApPage })));
const WorkQueuePage = lazy(() => import("./pages/WorkQueue").then((m) => ({ default: m.WorkQueuePage })));
const ReviewPage = lazy(() => import("./pages/Review").then((m) => ({ default: m.ReviewPage })));
const EvalsPage = lazy(() => import("./pages/Evals").then((m) => ({ default: m.EvalsPage })));
const DatasetsPage = lazy(() => import("./pages/Datasets").then((m) => ({ default: m.DatasetsPage })));
const ResearchPage = lazy(() => import("./pages/Research").then((m) => ({ default: m.ResearchPage })));

const ROLES: Role[] = ["viewer", "engineer", "approver", "admin"];

function WhoAmI() {
  const [me, setMe] = useState<Identity>(identity());
  const update = (patch: Partial<Identity>) => { const next = { ...me, ...patch }; setMe(next); setIdentity(next); };
  return (
    <div className="who">
      <label>Signed in as (demo)
        <input type="text" value={me.user} placeholder="Your name" onChange={(e) => update({ user: e.target.value })} aria-label="Your name" />
      </label>
      <label>Role
        <select value={me.role} onChange={(e) => update({ role: e.target.value as Role })} aria-label="Role">
          {ROLES.map((r) => <option key={r} value={r}>{r[0].toUpperCase() + r.slice(1)}</option>)}
        </select>
      </label>
    </div>
  );
}

export function App() {
  const [chat, setChat] = useState(false);
  const health = useAsync(() => gql<{ health: { llm_provider: string; circuit: string; models?: { small: string } } }>(`{ health }`)
    .then((d) => d.health));
  const pending = useAsync(() => gql<{ approvals: Approval[] }>(Q.approvals, { s: "pending" }).then((d) => d.approvals.length), []);
  useEffect(() => {
    const t = setInterval(() => void pending.reload(), 5000);
    return () => clearInterval(t);
  }, [pending.reload]);
  return (
    <div className={`shell ${chat ? "chat-open" : ""}`}>
      <nav className="rail" aria-label="Main">
        <div className="brand">Aero<span>Nexus</span><small>Aerospace documents and data to traceable decisions</small></div>
        <div className="nav">
          <NavLink to="/" end>Overview</NavLink>
          <div className="group">Workflows</div>
          <NavLink to="/change">Change orders</NavLink>
          <NavLink to="/pdm">Predictive maintenance</NavLink>
          <NavLink to="/safety">Safety analytics</NavLink>
          <NavLink to="/quote">Drawing to quote</NavLink>
          <NavLink to="/mro">Directive tracker</NavLink>
          <NavLink to="/ap">Invoice matching</NavLink>
          <div className="group">Oversight</div>
          <NavLink to="/queue">Work queue</NavLink>
          <NavLink to="/review" onClick={() => void pending.reload()}>Approvals <small>{pending.data || ""}</small></NavLink>
          <NavLink to="/evals">Evaluation</NavLink>
          <NavLink to="/data">Data and rules</NavLink>
          <NavLink to="/research">Research basis</NavLink>
        </div>
        <div className="rail-foot">
          <WhoAmI />
          <div style={{ marginTop: 10 }}>
            {health.error ? <><span className="dot down" />API offline</> :
              health.data && <><span className={`dot ${health.data.circuit === "open" ? "down" : ""}`} />Model: {health.data.llm_provider}
                {health.data.models && health.data.llm_provider !== "mock" ? ` · ${health.data.models.small}` : ""}
                {health.data.circuit === "open" ? " (paused, rules only)" : ""}</>}
          </div>
        </div>
      </nav>
      <main>
        <Suspense fallback={<div className="skeleton" style={{ width: 240 }} />}>
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/change" element={<ChangeOrdersPage />} />
            <Route path="/pdm" element={<PdmPage />} />
            <Route path="/safety" element={<SafetyPage />} />
            <Route path="/quote" element={<QuotePage />} />
            <Route path="/mro" element={<MroPage />} />
            <Route path="/ap" element={<ApPage />} />
            <Route path="/queue" element={<WorkQueuePage />} />
            <Route path="/review" element={<ReviewPage />} />
            <Route path="/evals" element={<EvalsPage />} />
            <Route path="/data" element={<DatasetsPage />} />
            <Route path="/research" element={<ResearchPage />} />
          </Routes>
        </Suspense>
      </main>
      {chat ? <Suspense fallback={null}><ChatPanel onClose={() => setChat(false)} /></Suspense> :
        <button className="btn primary chat-toggle" onClick={() => setChat(true)}>Ask about results</button>}
    </div>
  );
}
