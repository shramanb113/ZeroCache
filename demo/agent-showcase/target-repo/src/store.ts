export interface Link {
  id: string;
  url: string;
  createdAt: number;
}

const links = new Map<string, Link>();

export function listLinks(): Link[] {
  return [...links.values()];
}

export function addLink(url: string): Link {
  const id = Math.random().toString(36).slice(2, 10);
  const link: Link = { id, url, createdAt: Date.now() };
  links.set(id, link);
  return link;
}

export function deleteLink(id: string): boolean {
  return links.delete(id);
}

export function _reset(): void {
  links.clear();
}
