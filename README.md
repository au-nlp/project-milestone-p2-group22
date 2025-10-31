# Natural Language Processing

Group 22, NLP 2025, Fall.

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hgNAtOO3)

Repository for the course project.

The group members: Berg, Mikkel Skafsgaard; Jehøj-Krogager, Nikolaj; Yildirim, Michel.

The dataset being used: SMOL

Department of Computer Science, Aarhus University, 2025.

We have provided a `env.yaml` to setup a proper Python environment to use in this project. To create it run

```sh
$ conda env create -f env.yaml
```

## Update mirror

We develop on the AU GitLab instance. This course requires our work to be on
GitHub, so we have set up a mirror, but we need to manually push the changes.
We do not have access to the settings, such that it could be automated. On
e.g. Forgejo that is possible. You should be in the "extra" repository (see
[docs.github.com](https://docs.github.com/en/repositories/creating-and-managing-repositories/duplicating-a-repository#mirroring-a-repository-in-another-location)
 for more information), which
in my case is `nlp_backup` and run

```
$ git fetch -p origin
$ git push --mirror
```
