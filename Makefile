.PHONY: tags

tags:
	@awk '/^tags:/{f=1;next} f&&/^- /{gsub(/^- /,"");print;next} f{f=0}' content/posts/*.md 2>/dev/null | sort | uniq -c | sort -rn | awk '{printf "  %-20s (%d)\n", $$2, $$1}'
