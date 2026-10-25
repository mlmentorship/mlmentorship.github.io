import registry from '../data/learningMechanisms.json';
import { fromMarkdown } from 'mdast-util-from-markdown';
import type { RootContent, PhrasingContent, ListItem } from 'mdast';

export type LearningPiece = (typeof registry.pieces)[number];
const pieces = new Map(registry.pieces.map((piece) => [piece.slug, piece]));
const families = new Map(registry.families.map((family) => [family.id, family]));

function resolve(slugs: string[]): LearningPiece[] {
  return [...new Set(slugs)].flatMap((slug) => {
    const piece = pieces.get(slug);
    return piece ? [piece] : [];
  });
}

export function getLearningContext(slug: string) {
  const article = pieces.get(slug);
  if (!article) return undefined;
  const family = families.get(article.family);
  const owner = pieces.get(article.owner);
  if (!family || !owner) return undefined;
  return {
    article, family, owner,
    required: resolve(article.required),
    applications: resolve(family.members.filter((member) => member !== slug && member !== owner.slug)),
    related: resolve(article.related.filter((related) => related !== slug && related !== owner.slug && !article.required.includes(related) && !family.members.includes(related))),
  };
}

export function getMechanismFamilies() {
  return registry.families.map((family) => ({ ...family, owner: pieces.get(family.owner)! }));
}

export const mechanismReadingOrder = registry.readingOrder;

export function getColdPrompt(source: string): string | undefined {
  type ContentNode = RootContent | PhrasingContent | ListItem;
  function text(node: ContentNode): string {
    if (node.type === 'html') return '';
    if ('value' in node) return node.value;
    if ('children' in node) return node.children.map(text).join(node.type === 'list' ? '\n' : '');
    return '';
  }
  const body = source.replace(/^---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/, '');
  const nodes = fromMarkdown(body).children;
  const start = nodes.findIndex((node) => node.type === 'heading' && node.depth === 2 && text(node) === 'Problem');
  if (start < 0) {
    const quoted = nodes.find((node) => node.type === 'blockquote');
    return quoted ? text(quoted).trim() || undefined : undefined;
  }
  const remainder = nodes.slice(start + 1);
  const end = remainder.findIndex((node) => node.type === 'heading' && node.depth === 2);
  return remainder.slice(0, end < 0 ? remainder.length : end).map(text).filter(Boolean).join('\n\n').trim() || undefined;
}