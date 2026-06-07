/* jobctl i18n — client-side EN / 中文 instant toggle (no reload) */
"use strict";

const I18N = {
  en: {
    /* nav */
    "nav.dashboard":   "Dashboard",
    "nav.runs":        "Runs",

    /* dashboard sections */
    "section.servers":   "Servers",
    "section.runs":      "Runs",
    "runs.total":        "total",
    "col.run":           "Run",
    "queue.yours":       "yours",
    "queue.cluster":     "cluster",
    "queue.running":     "running",
    "queue.pending":     "pending",
    "queue.waiting":     "waiting",
    "queue.idle.cpu":    "idle cpu",
    "dash.eyebrow":      "research run gateway",
    "dash.live":         "live · htmx",
    "dash.managed.active": "managed active",
    "dash.scheduler.visible": "scheduler visible",
    "dash.attention":    "attention",
    "dash.servers.sub":  "queue, capacity and policy signals",
    "dash.run.lanes":    "Run lanes",
    "dash.runs.sub":     "jobctl records plus scheduler-only jobs",
    "dash.config":       "Configuration",
    "dash.config.sub":   "resolved paths, daemon settings and policy defaults",
    "config.paths":      "Paths",
    "config.runtime":    "Runtime",
    "config.policies":   "Default Policies",
    "config.servers":    "Configured Servers",
    "config.no.policies": "No default policies configured.",
    "config.no.servers": "No servers configured in cluster.yaml.",
    "config.snippet":    "TOML shape",
    "policy.label":      "policy",
    "policy.keep":       "keep",
    "policy.idle":       "idle",
    "policy.kernel.slots": "kernel slot(s)",
    "policy.capacity.unknown": "capacity unknown",

    /* bucket labels */
    "bucket.running":   "Running",
    "bucket.queued":    "Queued",
    "bucket.stuck":     "Stuck",
    "bucket.weak":      "Weak Signal",
    "bucket.completed": "Completed",
    "bucket.failed":    "Failed",

    /* empty states */
    "empty.running":   "No running runs.",
    "empty.queued":    "No queued runs.",
    "empty.stuck":     "No stuck runs.",
    "empty.weak":      "No weak signal runs.",
    "empty.completed": "No completed runs.",
    "empty.failed":    "No failed runs.",
    "empty.servers":   "No servers registered.",
    "empty.artifacts": "No artifacts.",
    "empty.runs":      "No runs yet.",
    "empty.params":    "No parameters defined.",
    "empty.obs":       "No observation card yet.",
    "empty.contract":  "No expectation contract yet.",
    "empty.criteria":  "No criteria.",

    /* run detail field labels */
    "label.jobfile":   "JobFile",
    "label.backend":   "Backend",
    "label.server":    "Server",
    "label.submitted": "Submitted",
    "label.finished":  "Finished",
    "label.params":    "Parameters",
    "label.slurm":     "SLURM submission",
    "label.resources": "Resources",

    /* log section */
    "section.logs":    "Logs",
    "label.stdout":    "stdout",
    "label.stderr":    "stderr",
    "no.stdout":       "(no stdout)",
    "no.stderr":       "(no stderr)",

    /* artifacts */
    "section.artifacts": "Artifacts",

    /* observation */
    "section.obs":       "Observation",
    "label.next":        "Next",
    "label.evidence":    "Key evidence",

    /* expectation contract */
    "section.contract":  "Expectation Contract",
    "col.criterion":     "Criterion",
    "col.kind":          "Kind",
    "col.status":        "Status",
    "col.strength":      "Strength",
    "col.result":        "Result",
    "col.detail":        "Detail",

    /* jobfile detail */
    "section.command":   "Command Template",
    "section.schema":    "Parameters Schema",
    "section.metadata":  "Metadata",
    "section.history":   "Historical Runs",
    "col.runid":         "Run ID",
    "col.state":         "State",
    "col.match":         "Match",
    "col.server":        "Server",
    "col.backend":       "Backend",
    "col.submitted":     "Submitted",
    "col.params":        "Params",
    "col.name":          "Name",
    "col.type":          "Type",
    "col.default":       "Default",
    "col.required":      "Required",
    "meta.id":           "ID",
    "meta.source":       "Source",
    "meta.hash":         "Hash",
    "meta.created":      "Created",
    "meta.artifacts":    "Artifacts",

    /* footer */
    "footer.label":      "jobctl",

    /* misc */
    "label.polling":     "polling…",
    "rows.x.cols":       "rows × cols",
    "label.bytes":       "bytes",
    "contract.criteria": "criteria",
    "contract.source":   "source",
    "contract.v":        "Contract v",
  },

  zh: {
    /* nav */
    "nav.dashboard":   "仪表板",
    "nav.runs":        "运行",

    /* dashboard sections */
    "section.servers":   "服务器",
    "section.runs":      "运行列表",
    "runs.total":        "总计",
    "col.run":           "运行",
    "queue.yours":       "我的",
    "queue.cluster":     "全集群",
    "queue.running":     "运行中",
    "queue.pending":     "排队中",
    "queue.waiting":     "等待中",
    "queue.idle.cpu":    "空闲 CPU",
    "dash.eyebrow":      "研究任务网关",
    "dash.live":         "实时 · htmx",
    "dash.managed.active": "托管活跃",
    "dash.scheduler.visible": "调度器可见",
    "dash.attention":    "需要关注",
    "dash.servers.sub":  "队列、容量与策略信号",
    "dash.run.lanes":    "运行通道",
    "dash.runs.sub":     "jobctl 记录与调度器任务",
    "dash.config":       "配置",
    "dash.config.sub":   "已解析路径、daemon 设置与默认策略",
    "config.paths":      "路径",
    "config.runtime":    "运行时",
    "config.policies":   "默认策略",
    "config.servers":    "已配置服务器",
    "config.no.policies": "未配置默认策略。",
    "config.no.servers": "cluster.yaml 中未配置服务器。",
    "config.snippet":    "TOML 结构",
    "policy.label":      "策略",
    "policy.keep":       "保留",
    "policy.idle":       "空闲",
    "policy.kernel.slots": "kernel 槽位",
    "policy.capacity.unknown": "容量未知",

    /* bucket labels */
    "bucket.running":   "运行中",
    "bucket.queued":    "排队中",
    "bucket.stuck":     "卡住",
    "bucket.weak":      "弱信号",
    "bucket.completed": "已完成",
    "bucket.failed":    "失败",

    /* empty states */
    "empty.running":   "暂无运行中的任务。",
    "empty.queued":    "暂无排队中的任务。",
    "empty.stuck":     "暂无卡住的任务。",
    "empty.weak":      "暂无弱信号任务。",
    "empty.completed": "暂无已完成的任务。",
    "empty.failed":    "暂无失败的任务。",
    "empty.servers":   "未注册服务器。",
    "empty.artifacts": "无产出物。",
    "empty.runs":      "暂无运行记录。",
    "empty.params":    "未定义参数。",
    "empty.obs":       "暂无观察卡片。",
    "empty.contract":  "暂无期望合约。",
    "empty.criteria":  "暂无标准。",

    /* run detail field labels */
    "label.jobfile":   "任务文件",
    "label.backend":   "后端",
    "label.server":    "服务器",
    "label.submitted": "提交时间",
    "label.finished":  "完成时间",
    "label.params":    "参数",
    "label.slurm":     "SLURM 提交参数",
    "label.resources": "资源",

    /* log section */
    "section.logs":    "日志",
    "label.stdout":    "标准输出",
    "label.stderr":    "标准错误",
    "no.stdout":       "（无标准输出）",
    "no.stderr":       "（无标准错误）",

    /* artifacts */
    "section.artifacts": "产出物",

    /* observation */
    "section.obs":       "观察",
    "label.next":        "下一步",
    "label.evidence":    "关键依据",

    /* expectation contract */
    "section.contract":  "期望合约",
    "col.criterion":     "标准",
    "col.kind":          "类型",
    "col.status":        "状态",
    "col.strength":      "强度",
    "col.result":        "结果",
    "col.detail":        "详情",

    /* jobfile detail */
    "section.command":   "命令模板",
    "section.schema":    "参数模式",
    "section.metadata":  "元数据",
    "section.history":   "历史运行",
    "col.runid":         "运行 ID",
    "col.state":         "状态",
    "col.match":         "匹配",
    "col.server":        "服务器",
    "col.backend":       "后端",
    "col.submitted":     "提交时间",
    "col.params":        "参数",
    "col.name":          "名称",
    "col.type":          "类型",
    "col.default":       "默认值",
    "col.required":      "必填",
    "meta.id":           "ID",
    "meta.source":       "来源",
    "meta.hash":         "哈希",
    "meta.created":      "创建时间",
    "meta.artifacts":    "产出物",

    /* footer */
    "footer.label":      "jobctl",

    /* misc */
    "label.polling":     "轮询中…",
    "rows.x.cols":       "行 × 列",
    "label.bytes":       "字节",
    "contract.criteria": "条标准",
    "contract.source":   "来源",
    "contract.v":        "合约 v",
  },
};

(function () {
  const STORAGE_KEY = "jobctl_lang";
  let _lang = localStorage.getItem(STORAGE_KEY) || "en";

  function applyLang(lang) {
    _lang = lang;
    localStorage.setItem(STORAGE_KEY, lang);

    const dict = I18N[lang] || I18N["en"];

    // Update all [data-i18n] elements
    document.querySelectorAll("[data-i18n]").forEach(function (el) {
      const key = el.getAttribute("data-i18n");
      if (dict[key] !== undefined) {
        el.textContent = dict[key];
      }
    });

    // Update toggle button active state
    document.querySelectorAll(".lang-btn").forEach(function (btn) {
      if (btn.dataset.lang === lang) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });
  }

  function init() {
    // Wire toggle buttons
    document.querySelectorAll(".lang-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        applyLang(btn.dataset.lang);
      });
    });

    // Apply current language
    applyLang(_lang);

    // Re-apply after HTMX swaps partials
    document.body.addEventListener("htmx:afterSwap", function () {
      applyLang(_lang);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
