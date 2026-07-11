export type SourceTier = 'Official' | 'Method-based report' | 'First-person report';

export interface FrontierSource {
  label: string;
  href: string;
  tier: SourceTier;
  checkedOn: string;
  supports: string;
}

export interface FrontierLabProcess {
  id: string;
  lab: string;
  bottomLine: string;
  confirmed: string[];
  reported: string[];
  aiPolicy: string;
  prep: Array<{ label: string; href: string }>;
  caution: string;
  sources: FrontierSource[];
}

const checkedOn = 'July 11, 2026';

export const FRONTIER_LAB_PROCESSES: FrontierLabProcess[] = [
  {
    id: 'openai',
    lab: 'OpenAI',
    bottomLine: 'Prepare for a role-specific work sample and a long final loop. Do not assume one standard engineering or research process.',
    confirmed: [
      'Application and resume review, then an introductory recruiter or hiring-manager call.',
      'One or more skills-based assessments. Official examples include pair coding, take-home projects, and technical tests.',
      'A final loop of roughly 4 to 6 hours with 4 to 6 people over 1 or 2 days.',
      'Engineering work is judged on solution design, code quality, performance, tests, communication, and collaboration.',
    ],
    reported: [
      'Current engineer reports add practical multi-part coding, architecture or system design, a technical project presentation, and separate leadership or collaboration discussions.',
      'A June 2026 report describes a beta agentic-coding round in an existing codebase. Treat this as a pilot, not a universal round.',
      'Some research-engineering reports include ML debugging, statistics, probability, or information-theory work. Team scope matters more than the company label.',
    ],
    aiPolicy: 'The official guide does not publish one universal live-interview AI rule. A current method-based report says AI is restricted except in an explicitly agentic pilot. Follow the recruiter instructions for the exact round.',
    prep: [
      { label: 'Agentic ML codebase lab', href: '/prep/labs/agentic-codebase/' },
      { label: 'Technical presentation', href: '/prep/presentation/' },
      { label: 'LLM inference design', href: '/questions/design-production-llm-inference-service/' },
      { label: 'Math oral', href: '/prep/labs/math-oral/' },
    ],
    caution: 'OpenAI explicitly says experiences vary by team. Ask for the assessment format, editor, AI policy, presentation requirement, and domain before choosing a drill.',
    sources: [
      { label: 'OpenAI interview guide', href: 'https://openai.com/interview-guide/', tier: 'Official', checkedOn, supports: 'Stages, assessment examples, final-loop length, and engineering criteria.' },
      { label: 'interviewing.io OpenAI process', href: 'https://interviewing.io/openai-interview-questions', tier: 'Method-based report', checkedOn, supports: 'Practical coding, presentation, system design, and agentic-coding pilot.' },
      { label: 'Frontier Evals and Environments role', href: 'https://openai.com/careers/research-engineer-frontier-evals-and-environments-san-francisco/', tier: 'Official', checkedOn, supports: 'Current work in RL environments, graders, synthetic data, and evaluation systems.' },
    ],
  },
  {
    id: 'anthropic',
    lab: 'Anthropic',
    bottomLine: 'Expect practical Python, explicit values evaluation, and team-dependent work samples. Performance and research tracks can look nothing like a standard SWE loop.',
    confirmed: [
      'Technical interviews use live coding tools such as Colab and CodeSignal. Looking up syntax is allowed, but fluency still matters.',
      'AI is not allowed in take-homes or live interviews unless the instructions explicitly permit it.',
      'The performance team has used timed take-homes based on a simulated accelerator and now publishes the original challenge.',
      'The Fellows process includes an application and reference check, technical assessments and interviews, and a research discussion.',
    ],
    reported: [
      'A June 2026 report describes a progressive CodeSignal task, hiring-manager project depth, practical coding, system design, and a standalone company-values round for many SWE candidates.',
      'A 2025 Fellows candidate reported a short research brainstorm and a longer black-box model investigation followed by a presentation.',
      'Round count and composition vary substantially across software, research, safety, and performance roles.',
    ],
    aiPolicy: 'Complete take-homes and live interviews without AI unless Anthropic explicitly says otherwise. The public performance challenge is an exception where AI use is allowed, but weakening tests invalidates the result.',
    prep: [
      { label: 'Black-box research lab', href: '/prep/labs/black-box/' },
      { label: 'Accelerator performance lab', href: '/prep/labs/accelerator/' },
      { label: 'Values and mission practice', href: '/prep/values/' },
      { label: 'Broken training lab', href: '/prep/labs/broken-training/' },
    ],
    caution: 'Do not infer a full-time Anthropic loop from the Fellows process or the performance team challenge. Use each as evidence for a format only when the recruiter confirms it.',
    sources: [
      { label: 'Anthropic careers', href: 'https://www.anthropic.com/careers', tier: 'Official', checkedOn, supports: 'Live coding tools and hiring philosophy.' },
      { label: 'Candidate AI guidance', href: 'https://www.anthropic.com/candidate-ai-guidance', tier: 'Official', checkedOn, supports: 'AI rules for applications, take-homes, and live interviews.' },
      { label: 'AI-resistant technical evaluations', href: 'https://www.anthropic.com/engineering/AI-resistant-technical-evaluations', tier: 'Official', checkedOn, supports: 'Performance take-home design, simulated accelerator, profiling, and time limits.' },
      { label: 'Anthropic Fellows Program', href: 'https://job-boards.greenhouse.io/anthropic/jobs/5023394008', tier: 'Official', checkedOn, supports: 'References, technical assessment, research discussion, and current workstreams.' },
      { label: 'interviewing.io Anthropic process', href: 'https://interviewing.io/anthropic-interview-questions', tier: 'Method-based report', checkedOn, supports: 'Current reported SWE stages, CodeSignal format, values round, and system design.' },
    ],
  },
  {
    id: 'deepmind',
    lab: 'Google DeepMind',
    bottomLine: 'The process is role-specific, but Research Engineer preparation still needs executable coding, mathematical ML depth, model design, and a defensible project history.',
    confirmed: [
      'A 30-minute recruiter introduction, with a possible hiring-manager interview.',
      'Two or three skills interviews calibrated to the role.',
      'Final conversations with team leads and leadership through the lens of team plans, culture, mission, and values.',
      'Candidates receive role-specific preparation because the exact steps differ by position.',
    ],
    reported: [
      'A June 2026 synthesis of candidate reports describes two executable coding rounds followed by mathematical ML fundamentals and ML design for many Research Engineer loops.',
      'Reported ML prompts reward derivation and intuition, not definition recall. Constraints often change after the first answer.',
      'General algorithms remain a real requirement for some roles, but this site deliberately leaves that curriculum to dedicated resources.',
    ],
    aiPolicy: 'The public careers overview does not state one universal AI policy. Use the role-specific instructions sent by recruiting.',
    prep: [
      { label: 'Math oral', href: '/prep/labs/math-oral/' },
      { label: 'ML implementation set', href: '/prep/labs/implementation/' },
      { label: 'Research simulation', href: '/prep/simulations/#research-engineer' },
      { label: 'Paper critique', href: '/questions/critique-ml-paper/' },
    ],
    caution: 'Do not substitute generic Google interview guidance for DeepMind role instructions. Research Engineer, Research Scientist, and product-facing ML roles use different mixes.',
    sources: [
      { label: 'Google DeepMind careers', href: 'https://deepmind.google/careers/', tier: 'Official', checkedOn, supports: 'Initial, skills, final, and decision stages.' },
      { label: '2026 Research Engineer synthesis', href: 'https://igotanoffer.com/en/advice/google-deepmind-research-engineer-interview', tier: 'Method-based report', checkedOn, supports: 'Reported coding, ML fundamentals, ML design, and team interviews.' },
    ],
  },
  {
    id: 'meta',
    lab: 'Meta AI',
    bottomLine: 'AI-assisted coding and design are now first-class formats for selected roles. The signal is control, review, debugging, and judgment, not prompt volume.',
    confirmed: [
      'Many interviews now include an AI assistant built into the interview environment, and candidates are expected to use it.',
      'The assistant is built into CoderPad and offers Claude, ChatGPT, Gemini, and Meta models.',
      'AI-native design interviews use Mermaid Markdown in CoderPad with the same assistant available.',
      'Meta recommends practicing in the provided environment and getting comfortable reading, debugging, and extending existing code.',
    ],
    reported: [
      'A 2026 report describes a 60-minute, multi-file onsite round that replaces one coding interview for some roles.',
      'Candidates are judged on planning, codebase navigation, bounded delegation, critical review, testing, and explanation.',
      'This format does not remove the need for unaided algorithmic or role-specific technical rounds.',
    ],
    aiPolicy: 'Use only the assistant inside the authorized interview environment. Outside AI tools are not permitted. Select the model in the environment and remain responsible for every change.',
    prep: [
      { label: 'Agentic ML codebase lab', href: '/prep/labs/agentic-codebase/' },
      { label: 'Agentic interview question', href: '/questions/agentic-ml-codebase-interview/' },
      { label: 'ML system design', href: '/questions/design-production-llm-inference-service/' },
    ],
    caution: 'Meta says selected roles use the format. Confirm whether your loop includes AI-native coding, AI-native design, both, or neither.',
    sources: [
      { label: 'Meta hiring process and AI FAQ', href: 'https://www.metacareers.com/hiring-process/', tier: 'Official', checkedOn, supports: 'AI expectations, models, languages, design environment, and preparation.' },
      { label: 'AI-assisted coding report', href: 'https://interviewing.io/blog/how-to-use-ai-in-meta-s-ai-assisted-coding-interview-with-real-prompts-and-examples', tier: 'Method-based report', checkedOn, supports: 'Reported duration, onsite placement, and multi-file task shape.' },
    ],
  },
  {
    id: 'xai',
    lab: 'xAI',
    bottomLine: 'The public process is sparse but unusually direct: technical staff screen the application, ask short technical questions, and then go deep on relevant expertise.',
    confirmed: [
      'Applications are evaluated by technical team members rather than recruiters for assessment.',
      'The screening interview covers background, fit, and short technical questions.',
      'Technical interviews examine complex problem solving and critical thinking in the candidate’s domain.',
      'Applications ask for a statement of exceptional work in 100 words or fewer.',
    ],
    reported: [
      'Public evidence is not strong enough to publish a stable ML or Research Engineer round-by-round recipe.',
      'Use the job description and recruiter guidance to identify model training, infrastructure, product, or domain depth.',
    ],
    aiPolicy: 'No universal public interview AI rule was found. Ask before using any tool that is not explicitly provided.',
    prep: [
      { label: 'Exceptional-work worksheet', href: '/guides/frontier-lab-proof-of-work-and-references/' },
      { label: 'Technical presentation', href: '/prep/presentation/' },
      { label: 'Training systems', href: '/questions/train-100b-model/' },
    ],
    caution: 'Avoid SEO question lists that claim exact xAI prompts without a disclosed source. The official process supports format-level preparation, not a question dump.',
    sources: [
      { label: 'xAI careers', href: 'https://x.ai/careers', tier: 'Official', checkedOn, supports: 'Technical review, screening, technical interviews, and application artifact.' },
      { label: 'Model Training role', href: 'https://job-boards.greenhouse.io/xai/jobs/5086324007', tier: 'Official', checkedOn, supports: 'Current role scope and exceptional-work application prompt.' },
    ],
  },
];
