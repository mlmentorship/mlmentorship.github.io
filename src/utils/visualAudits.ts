export interface VisualAuditSummary {
  slug: string;
  status: 'implemented' | 'planned' | 'no-visual' | 'unreviewed';
  medium: string;
  learningObjective: string;
  visualIds: string[];
}

interface VisualAuditRecord {
  slug?: string;
  status?: VisualAuditSummary['status'];
  medium?: string;
  learningObjective?: string;
  implementation?: {
    visualIds?: string[];
  };
}

const auditModules = import.meta.glob('../../data/visual-audits/*.json', {
  eager: true,
  import: 'default',
}) as Record<string, VisualAuditRecord>;

const auditsBySlug = new Map<string, VisualAuditSummary>();

for (const [path, audit] of Object.entries(auditModules)) {
  const filenameSlug = path.split('/').at(-1)?.replace(/\.json$/, '') ?? '';
  const slug = audit.slug ?? filenameSlug;
  if (!slug || !audit.status) continue;

  auditsBySlug.set(slug, {
    slug,
    status: audit.status,
    medium: audit.medium ?? 'none',
    learningObjective: audit.learningObjective ?? '',
    visualIds: audit.implementation?.visualIds ?? [],
  });
}

export function getVisualAudit(slug: string): VisualAuditSummary | undefined {
  return auditsBySlug.get(slug);
}

export function getImplementedVisualAudit(slug: string): VisualAuditSummary | undefined {
  const audit = getVisualAudit(slug);
  return audit?.status === 'implemented' && audit.visualIds.length > 0 ? audit : undefined;
}