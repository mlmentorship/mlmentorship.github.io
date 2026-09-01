function attributeValues(source, name) {
  const pattern = new RegExp(`\\b${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)')`, 'g');
  return [...source.matchAll(pattern)].map((match) => match[1] ?? match[2]);
}

export function unresolvedAccessibilityReferences(source) {
  const idCounts = new Map();
  for (const id of attributeValues(source, 'id')) {
    idCounts.set(id, (idCounts.get(id) ?? 0) + 1);
  }

  const problems = [];
  const referencedIds = new Set();
  for (const attribute of ['aria-labelledby', 'aria-describedby']) {
    for (const value of attributeValues(source, attribute)) {
      for (const id of value.trim().split(/\s+/).filter(Boolean)) {
        referencedIds.add(id);
        const count = idCounts.get(id) ?? 0;
        if (count === 0) problems.push(`${attribute} references missing #${id}`);
        if (count > 1) problems.push(`${attribute} references duplicate #${id}`);
      }
    }
  }

  for (const [id, count] of idCounts) {
    if (count > 1 && !referencedIds.has(id)) problems.push(`duplicate id #${id}`);
  }
  return problems;
}
