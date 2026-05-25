-- Optional Supabase schema for active learning feedback
create table if not exists query_feedback (
  id bigint generated always as identity primary key,
  query text not null,
  predicted_label text not null check (predicted_label in ('google', 'perplexity')),
  confidence float not null default 0,
  correct boolean not null default true,
  created_at timestamptz not null default now()
);

create index if not exists idx_query_feedback_created on query_feedback (created_at desc);
