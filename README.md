# SteLLaFuzz
SteLLaFuzz is a structure-guided fuzzing framework that leverages large language models (LLMs) to generate protocol-aware seed inputs for testing network protocol implementations. Unlike conventional fuzzers that rely on random mutations or fixed dictionaries, SteLLaFuzz automatically extracts message types, structural formats, and valid sequences from raw seed messages, enabling it to generate semantically valid and structurally diverse inputs.

---

## 🚀 Quick-Start Checklist

| Step                       | Command                                                  | Notes                                                                                                                                  |
| -------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **1 · Install tooling**    | `./deps.sh`                                              | Installs Docker CE, Python 3, *pandas*, *matplotlib*, and helper scripts.([Docker Documentation][3], [Pandas][4], [matplotlib.org][5]) |
| **2 · Build images**       | `KEY=<OPENAI_API_KEY> ./setup.sh`                        | Requires an OpenAI API key.([OpenAI][6])                                                                                        |
| **3 · Launch fuzzing run** | `./run.sh <n-containers> <minutes> <subjects> <fuzzers>` | e.g. `./run.sh 1 300 pure-ftpd stellafuzz`.                                                                                            |
| **4 · Inspect coverage**   | `./analyze.sh <subjects> [minutes]`                      | Produces CSV + PNG reports under `res_<subject>-<timestamp>/`.                                                                         |
| **5 · Tidy workspace**     | `./clean.sh`                                             | Removes containers and temporary logs.                                                                                                 |

> **Time budget:** full benchmark build (\~13 subjects × 3 fuzzers) ≈ 60 minuites.

---

## 📂 Repository Layout (top level)

```
SteLLaFuzz
├── aflnet: a modified version of AFLNet which outputs states and state transitions 
├── analyse.sh: analysis script 
├── benchmark: a modified version of ProfuzzBench
│   └── subjects/<subject>
│       ├── LLM: LLM-related files of SteLLaFuzz
│       ├── utility: utility file of SteLLaFuzz
│       └── stellafuzz.py: SteLLaFuzz implementation for the subject
├── clean.sh: clean script
├── ChatAFL: the source code of ChatAFL
├── SteLLaFuzz: the source code of SteLLaFuzz, same as aflnet
├── deps.sh: the script to install dependencies, asks for the password when executed
├── README: this file
├── run.sh: the execution script to run fuzzers on subjects and collect data
└── setup.sh: the preparation script to set up docker images
```

*Inside* each subject folder you will find:

* `LLM/` – prompt templates & few-shot examples
* `stellafuzz.py` – subject-specific harness
* `utility/` – mutation budget, timeout, retry constants

---

## 🔄 Typical Workflow

1. **Select subjects & fuzzers** – Any subset of the bundled FTP, SMTP, TLS, or SSH servers.
2. **Parallel execution** – `run.sh` schedules *n* Docker containers; each container isolates one fuzzer/subject pair, avoiding cross-interference.
3. **Monitoring** – Real-time stats are streamed through AFLNet's UI; detailed coverage curves are generated post-run by `analyze.sh`.
4. **Result triage** – Crashes and hangs are archived under `result-<subject>/crashes` ready for repro.

---

## 📊 Experimental Setup

The experiments in the SteLLaFuzz paper were conducted using the following execution script:

```bash
./run.sh 10 1440 dcmtk,dnsmasq,tinydtls,openssh,openssl,forked-daapd,bftpd,lightftp,proftpd,pure-ftpd,live555,exim,kamailio aflnet,chatafl,stellafuzz
```

---

## ⚙️ Tuning Options

| Variable          | Meaning                                               | Default       |
| ----------------- | ----------------------------------------------------- | ------------- |
| `MODEL`           | OpenAI model id used for grammar extraction           | `gpt-4o-mini` |
| `SEQUENCE_REPEAT` | How many alternative dialogues are generated per seed | `1`           |
| `LLM_RETRY`       | Fallback attempts before giving up on a prompt        | `3`           |

Edit `benchmark/subjects/<subject>/utility/utility.py` to experiment with more aggressive exploration or cheaper models.

---

## 📝 License

The code is released under the **Apache 2.0** license.([apache.org][8])

---

## 🤝 Acknowledgements

* AFLNet for the state-feedback engine.([GitHub][1])
* ChatAFL for demonstrating the potential of LLM-assisted mutation.([GitHub][2])
* ProFuzzBench for providing a reproducible evaluation bed.([GitHub][7])

[1]: https://github.com/aflnet/aflnet?utm_source=chatgpt.com "aflnet/aflnet: AFLNet: A Greybox Fuzzer for Network ... - GitHub"
[2]: https://github.com/ChatAFLndss/ChatAFL?utm_source=chatgpt.com "ChatAFLndss/ChatAFL: Large Language Model guided ... - GitHub"
[3]: https://docs.docker.com/engine/install/?utm_source=chatgpt.com "Install Docker Engine"
[4]: https://pandas.pydata.org/docs/?utm_source=chatgpt.com "pandas 2.2.3 documentation"
[5]: https://matplotlib.org/stable/tutorials/index.html?utm_source=chatgpt.com "Tutorials — Matplotlib 3.10.3 documentation"
[6]: https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key?utm_source=chatgpt.com "Where do I find my OpenAI API Key?"
[7]: https://github.com/profuzzbench/profuzzbench?utm_source=chatgpt.com "ProFuzzBench - A Benchmark for Stateful Protocol Fuzzing - GitHub"
[8]: https://www.apache.org/licenses/LICENSE-2.0?utm_source=chatgpt.com "Apache License, Version 2.0"
[9]: https://github.com/banesullivan/README?utm_source=chatgpt.com "How to write a good README - GitHub"
[10]: https://www.linkedin.com/pulse/protecting-your-open-source-project-comprehensive-guide-chan-meng-u77wc?utm_source=chatgpt.com "A Comprehensive Guide to Combating Code Plagiarism - LinkedIn"
[11]: https://www.wired.com/2010/08/write-your-readme-before-your-code?utm_source=chatgpt.com "Write Your README Before Your Code"
