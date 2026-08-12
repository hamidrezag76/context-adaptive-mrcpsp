class NetworkAnalysis:

    def __init__(self, project):

        self.project = project

    def forward_pass(self):
        order = self.project.topological_sort()

        for activity_id in order:

            activity = self.project.get_activity(activity_id)

            if activity.indegree == 0:

                activity.earliest_start = 0

            else:

                activity.earliest_start = max(

                    self.project.get_activity(pred).earliest_finish

                    for pred in activity.predecessors

                )

            if activity.selected_mode is None:

                duration = activity.modes[0].duration

            else:

                duration = activity.get_mode(
                    activity.selected_mode
                ).duration

            activity.earliest_finish = (

                activity.earliest_start
                + duration

            )

    def backward_pass(self):

        order = list(
            reversed(
                self.project.topological_sort()
            )
        )

        project_finish = max(

            activity.earliest_finish

            for activity in self.project.activities.values()

        )

        for activity_id in order:

            activity = self.project.get_activity(activity_id)

            if activity.outdegree == 0:

                activity.latest_finish = project_finish

            else:

                activity.latest_finish = min(

                    self.project.get_activity(s).latest_start

                    for s in activity.successors

                )

            if activity.selected_mode is None:

                duration = activity.modes[0].duration

            else:

                duration = activity.get_mode(
                    activity.selected_mode
                ).duration

            activity.latest_start = (

                activity.latest_finish
                - duration

            )

    def compute_total_float(self):

        for activity in self.project.activities.values():

            activity.total_float = (

                activity.latest_start

                - activity.earliest_start

            )

    def critical_path(self):

        return [

            activity.id

            for activity in self.project.ordered_activities

            if activity.total_float == 0

        ]

    def analyze(self):

        self.forward_pass()
        self.backward_pass()
        self.compute_total_float()
        self.critical = self.critical_path()