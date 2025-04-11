data("discoveries")


df_discoveries <- data.frame(
  Year = as.numeric(time(discoveries)),
  Discoveries = as.numeric(discoveries)
)

write.csv(df_discoveries, file = "discoveries.csv", row.names = FALSE)

