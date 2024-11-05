The annnotations are included in a json format. Here is an example question from BoolQ and its perturbations: 



```json
{
  "title": "Scotland at the FIFA World Cup",
  "paragraph": "Scotland have never advanced beyond the first round of the finals competition. They have missed out on progressing to the second round three times on goal difference: in 1974, when Brazil edged them out; in 1978, when the Netherlands progressed; and in 1982, when the USSR went through. Although Scotland have played at eight finals tournaments, they have qualified on nine occasions. The Scottish Football Association declined to participate in 1950 as Scotland were not the British champions.",
  "question": "have scotland ever been in the world cup final",
  "answer": "FALSE",
  "perturbed_questions": [
    {
      "perturbed_q": "have scotland ever been in the world cup final competition",
      "answer": "TRUE"
    },
    {
      "perturbed_q": "have scotland ever been in the world cup final competition beyond the first round",
      "answer": "FALSE"
    },
    {
      "perturbed_q": "have scotland ever been in eight world cup final competition",
      "answer": "TRUE"
    },
    {
      "perturbed_q": "have scotland ever been in nine world cup final competition",
      "answer": "FALSE"
    },
    {
      "perturbed_q": "have scotland participated in the 1950 world cup final competition",
      "answer": "FALSE"
    }
  ]
}
```

source: https://github.com/allenai/contrast-sets/tree/main/BoolQ