type StatusBadgeProps = {
  status: string;
};

export function StatusBadge({ status }: StatusBadgeProps) {
  return <span className={`status-badge status-badge-${status.toLowerCase()}`}>{status}</span>;
}
