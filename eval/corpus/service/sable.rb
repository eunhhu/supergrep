def ordered_rows(rows)
  rows.sort_by { |row| [-row[:weight], row[:path], row[:offset]] }
end

def safe_excerpt(text, limit = 72)
  visible = text.gsub(/[[:cntrl:]]/, "\\u{FFFD}")
  visible.length > limit ? "#{visible[0, limit]}…" : visible
end

def display_owner(owner)
  owner.to_s.capitalize
end
