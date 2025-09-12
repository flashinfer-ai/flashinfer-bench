const meta = {
  docs: {
    title: '',
    // Replace the folder item with its children in the sidebar
    display: 'children' as const,
    // Keep a page at /docs (uses docs/index.mdx)
    type: 'page' as const,
  },
}

export default meta
