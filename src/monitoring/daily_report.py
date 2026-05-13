"""Operational daily forecast report for non-technical consumers (Camila, Ana, Bruna)."""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from src.config.settings import get_settings
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)

TERMINAL_DISPLAY_NAMES: dict[str, str] = {
    "T1": "Terminal 1 — Santos Sul",
    "T2": "Terminal 2 — Santos Norte",
    "T3": "Terminal 3 — Guarujá",
    "T4": "Terminal 4 — São Vicente",
}

HORIZON_LABELS: dict[str, str] = {
    "D+1": "Amanhã (D+1)",
    "D+2": "Depois de Amanhã (D+2)",
    "D+3": "D+3",
}

_CSS = """
body { font-family: Arial, sans-serif; margin: 32px; background: #f8f9fa; color: #212529; }
h1 { color: #003366; margin-bottom: 4px; }
.subtitle { color: #6c757d; font-size: 14px; margin-bottom: 32px; }
.section { background: white; border-radius: 8px; padding: 24px; margin-bottom: 24px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
h2 { color: #003366; font-size: 18px; margin-top: 0; border-bottom: 2px solid #e9ecef;
     padding-bottom: 8px; }
table { border-collapse: collapse; width: 100%; }
th { background: #003366; color: white; padding: 10px 14px; text-align: left; font-size: 13px; }
td { padding: 10px 14px; border-bottom: 1px solid #e9ecef; font-size: 14px; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f1f3f5; }
.badge-green  { background: #d4edda; color: #155724; padding: 3px 10px; border-radius: 12px;
                font-size: 12px; font-weight: bold; }
.badge-yellow { background: #fff3cd; color: #856404; padding: 3px 10px; border-radius: 12px;
                font-size: 12px; font-weight: bold; }
.badge-red    { background: #f8d7da; color: #721c24; padding: 3px 10px; border-radius: 12px;
                font-size: 12px; font-weight: bold; }
.metric-box { display: inline-block; background: #e9ecef; border-radius: 6px; padding: 12px 20px;
              margin: 6px; text-align: center; min-width: 140px; }
.metric-value { font-size: 28px; font-weight: bold; color: #003366; }
.metric-label { font-size: 12px; color: #6c757d; margin-top: 4px; }
.alert-box { background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px;
             border-radius: 0 6px 6px 0; margin-bottom: 12px; }
.alert-box.critical { background: #f8d7da; border-left-color: #dc3545; }
.footer { color: #6c757d; font-size: 12px; margin-top: 32px; text-align: center; }
"""


def _drift_badge(detected: bool) -> str:
    if detected:
        return '<span class="badge-red">⚠ Desvio Detectado</span>'
    return '<span class="badge-green">✓ Normal</span>'


def _productivity_badge(value: float) -> str:
    if value >= 600:
        return f'<span class="badge-green">{value:.0f}</span>'
    if value >= 400:
        return f'<span class="badge-yellow">{value:.0f}</span>'
    return f'<span class="badge-red">{value:.0f}</span>'


def generate_daily_operational_report(
    predictions_df: pd.DataFrame,
    execution_date: str,
    model_version: str,
    monitoring_metrics: dict[str, float | int],
    output_name: str | None = None,
) -> Path:
    """Generate a human-readable HTML forecast report for operations teams."""
    settings = get_settings()
    output_name = output_name or f"operational_report_{execution_date}.html"
    output = settings.reports_dir / "daily" / output_name
    output.parent.mkdir(parents=True, exist_ok=True)

    data_drift = bool(monitoring_metrics.get("data_drift_detected", 0))
    pred_drift = bool(monitoring_metrics.get("prediction_drift_detected", 0))
    total_predictions = len(predictions_df)

    # Summary KPIs
    avg_productivity = predictions_df["predicted_productivity_tons_hour"].mean()
    min_productivity = predictions_df["predicted_productivity_tons_hour"].min()
    max_productivity = predictions_df["predicted_productivity_tons_hour"].max()

    # Build terminal × horizon pivot
    pivot_rows = ""
    for terminal_id, terminal_name in TERMINAL_DISPLAY_NAMES.items():
        terminal_df = predictions_df[predictions_df["terminal_id"] == terminal_id]
        if terminal_df.empty:
            continue
        cols = ""
        for horizon in ["D+1", "D+2", "D+3"]:
            h_df = terminal_df[terminal_df["forecast_horizon"] == horizon]
            if h_df.empty:
                cols += "<td>—</td>"
            else:
                val = float(h_df["predicted_productivity_tons_hour"].mean())
                cols += f"<td>{_productivity_badge(val)} t/h</td>"
        pivot_rows += f"<tr><td><strong>{html.escape(terminal_name)}</strong></td>{cols}</tr>"

    # Alerts section
    alerts_html = ""
    if data_drift:
        alerts_html += (
            '<div class="alert-box critical">'
            "<strong>⚠ Desvio de Dados Detectado</strong> — "
            "A distribuição dos dados operacionais de entrada desviou do padrão histórico. "
            "Verifique o relatório Evidently e acione o time de MLOps se o desvio persistir por 2 dias consecutivos."
            "</div>"
        )
    if pred_drift:
        alerts_html += (
            '<div class="alert-box">'
            "<strong>⚠ Desvio nas Previsões</strong> — "
            "A distribuição das previsões mudou significativamente. "
            "As previsões podem estar menos precisas. Monitore os resultados reais dos próximos dias."
            "</div>"
        )
    if not alerts_html:
        alerts_html = '<p style="color:#155724">✓ Nenhum alerta operacional para hoje.</p>'

    page = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <title>Previsão de Produtividade — {html.escape(execution_date)}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>📦 Previsão de Produtividade de Descarga</h1>
  <div class="subtitle">
    Data de execução: <strong>{html.escape(execution_date)}</strong> &nbsp;|&nbsp;
    Modelo versão: <strong>{html.escape(str(model_version))}</strong> &nbsp;|&nbsp;
    Total de previsões: <strong>{total_predictions}</strong>
  </div>

  <div class="section">
    <h2>Resumo Operacional</h2>
    <div class="metric-box">
      <div class="metric-value">{avg_productivity:.0f}</div>
      <div class="metric-label">Média Geral (t/h)</div>
    </div>
    <div class="metric-box">
      <div class="metric-value">{min_productivity:.0f}</div>
      <div class="metric-label">Mínimo (t/h)</div>
    </div>
    <div class="metric-box">
      <div class="metric-value">{max_productivity:.0f}</div>
      <div class="metric-label">Máximo (t/h)</div>
    </div>
    <div class="metric-box">
      <div class="metric-value">{_drift_badge(data_drift)}</div>
      <div class="metric-label">Qualidade dos Dados</div>
    </div>
  </div>

  <div class="section">
    <h2>Previsão por Terminal e Horizonte (tons/hora)</h2>
    <p style="font-size:13px;color:#6c757d">
      🟢 ≥ 600 t/h — Alta produtividade &nbsp;|&nbsp;
      🟡 400–599 t/h — Atenção &nbsp;|&nbsp;
      🔴 &lt; 400 t/h — Baixa produtividade
    </p>
    <table>
      <thead>
        <tr>
          <th>Terminal</th>
          <th>Amanhã (D+1)</th>
          <th>Depois de Amanhã (D+2)</th>
          <th>D+3</th>
        </tr>
      </thead>
      <tbody>{pivot_rows}</tbody>
    </table>
  </div>

  <div class="section">
    <h2>Alertas e Qualidade do Modelo</h2>
    {alerts_html}
  </div>

  <div class="section">
    <h2>Guia de Uso</h2>
    <table>
      <thead><tr><th>Persona</th><th>Como usar este relatório</th></tr></thead>
      <tbody>
        <tr>
          <td><strong>Camila (Operações)</strong></td>
          <td>Use os valores D+1 por terminal para definir escalas operacionais e prioridade de descarga.
              Valores 🔴 indicam terminais que podem precisar de reforço de equipe ou replanejamento.</td>
        </tr>
        <tr>
          <td><strong>Ana (Control Tower)</strong></td>
          <td>D+2 e D+3 sinalizam gargalos futuros. Se dois ou mais terminais estiverem 🔴 no mesmo horizonte,
              acione o plano de contingência de sequenciamento operacional.</td>
        </tr>
        <tr>
          <td><strong>Bruna (Comercial)</strong></td>
          <td>Valores 🟢 consistentes em D+1/D+2 indicam baixo risco de quebra contratual.
              Valores 🔴 com ⚠ desvio de dados devem ser considerados nas negociações de SLA.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="footer">
    Gerado automaticamente pelo pipeline MLOps — porto-produtividade &nbsp;|&nbsp;
    Modelo: {html.escape(str(model_version))} &nbsp;|&nbsp; {html.escape(execution_date)}
  </div>
</body>
</html>"""

    output.write_text(page, encoding="utf-8")
    LOGGER.info("Operational report written to %s", output)

    # Also write a CSV summary for Bruna / commercial tooling
    csv_output = settings.reports_dir / "daily" / f"forecast_summary_{execution_date}.csv"
    summary_rows = []
    for terminal_id, terminal_name in TERMINAL_DISPLAY_NAMES.items():
        terminal_df = predictions_df[predictions_df["terminal_id"] == terminal_id]
        for horizon in ["D+1", "D+2", "D+3"]:
            h_df = terminal_df[terminal_df["forecast_horizon"] == horizon]
            if h_df.empty:
                continue
            forecast_date = h_df["forecast_date"].iloc[0] if "forecast_date" in h_df.columns else ""
            summary_rows.append({
                "execution_date": execution_date,
                "terminal_id": terminal_id,
                "terminal_name": terminal_name,
                "forecast_horizon": horizon,
                "forecast_date": forecast_date,
                "predicted_productivity_tons_hour": round(float(h_df["predicted_productivity_tons_hour"].mean()), 2),
                "data_drift_detected": int(data_drift),
                "prediction_drift_detected": int(pred_drift),
                "model_version": model_version,
            })
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(csv_output, index=False)
        LOGGER.info("Forecast summary CSV written to %s", csv_output)

    return output
