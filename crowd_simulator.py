import bisect
import csv
import numpy as np


class Simulator:

    def __init__(self, num_people, num_questions, options_per_question, answers_per_question="single", sparsity=0):
        assert(sparsity >= 0 and sparsity <= 1)
        assert(num_people > 0)
        assert(num_questions > 0)
        assert(options_per_question > 0)
        assert(answers_per_question ==
               "single" or answers_per_question == "multiple")
        self.num_p = num_people
        self.num_q = num_questions
        self.options_per_q = options_per_question
        self.answers_per_q = answers_per_question
        self.sparsity = sparsity

        assert(sparsity == 0)  # temporary, will be removed later

    def generate_ground_truths(self):
        ground_truths = np.zeros((self.num_q, self.options_per_q))
        for i in range(0, self.num_q):
            if self.answers_per_q == "single":
                answer_question_i = np.random.randint(0, self.options_per_q)
                ground_truths[i][answer_question_i] = 1
            else:
                for j in range(0, self.options_per_q):
                    answer_qi_option_j = np.random.randint(0, 2)
                    ground_truths[i][j] = answer_qi_option_j
        return ground_truths

    def generate_data(self, types_of_people):
        assert(len(types_of_people) > 0)
        ground_truths = self.generate_ground_truths()
        total_sum_of_proportions = 100
        cumulative_proportions = self.get_cumulative_proportions(
            total_sum_of_proportions, types_of_people)
        data = np.zeros((self.num_p, self.num_q, self.options_per_q))
        for person in range(0, self.num_p):
            type_of_person = self.get_type_of_person(
                cumulative_proportions, total_sum_of_proportions)
            prob_of_person = self.get_prob_of_person(
                types_of_people[type_of_person])
            if self.answers_per_q == "single":
                for question in range(0, self.num_q):
                    correct_answer = self.get_correct_answer(
                        ground_truths, question)
                    answer_of_person = self.get_answer_of_person(
                        correct_answer, prob_of_person, self.options_per_q)
                    data[person][question][answer_of_person] = 1
            else:
                for question in range(0, self.num_q):
                    correct_answer = self.get_correct_answer(
                        ground_truths, question)
                    for option in range(0, self.options_per_q):
                        answer_of_person = self.get_answer_of_person(
                            correct_answer[option], prob_of_person, 2)
                        data[person][question][option] = answer_of_person
        return data, ground_truths

    def write_to_csv(self, file_name, data, mode="value", order_by="person"):
        assert(mode == "value" or mode == "vector")
        assert(order_by == "person" or order_by == "question")
        data = self.flatten_data(data, order_by, mode)
        with open(file_name, "w") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerows(data)

    def flatten_data(self, data, order_by, mode):
        result_list = []
        if len(np.shape(data)) != 3:
            if self.answers_per_q == "multiple" or (self.answers_per_q == "single" and mode == "vector"):
                for question in range(0, self.num_q):
                    result_list.append([question, data[question].tolist()])
            else:
                for question in range(0, self.num_q):
                    result_list.append(
                        [question, np.where(data[question] == 1)[0].tolist()])
        elif self.answers_per_q == "multiple" or (self.answers_per_q == "single" and mode == "vector"):
            if order_by == "person":
                for person in range(0, self.num_p):
                    for question in range(0, self.num_q):
                        result_list.append(
                            [person, question, data[person][question].tolist()])
            else:
                for question in range(0, self.num_q):
                    for person in range(0, self.num_p):
                        result_list.append(
                            [question, person, data[person][question].tolist()])
        else:
            if order_by == "person":
                for person in range(0, self.num_p):
                    for question in range(0, self.num_q):
                        result_list.append([person, question, np.where(
                            data[person][question] == 1)[0].tolist()])
            else:
                for question in range(0, self.num_q):
                    for person in range(0, self.num_p):
                        result_list.append([question, person, np.where(
                            data[person][question] == 1)[0].tolist()])
        flattened_list = self.flatten_list(result_list)
        return flattened_list

    def flatten_list(self, input_list):
        result_list = []
        for i in range(0, len(input_list)):
            result_list.append(self.flatten_list_element(input_list[i]))
        return result_list

    def flatten_list_element(self, input_list):
        result_list = []
        for i in range(0, len(input_list)):
            try:
                len(input_list[i])
                for j in range(0, len(input_list[i])):
                    result_list.append(int(input_list[i][j]))
            except:
                result_list.append(int(input_list[i]))
        return result_list

    def get_cumulative_proportions(self, total_sum_of_proportions, types_of_people):
        sum_of_proportions = 0
        for i in range(0, len(types_of_people)):
            sum_of_proportions += types_of_people[i].proportion
        assert(total_sum_of_proportions == sum_of_proportions)
        cumulative_proportions = np.zeros(len(types_of_people))
        cumulative_proportions[0] = types_of_people[0].proportion
        for i in range(1, len(types_of_people)):
            cumulative_proportions[i] = cumulative_proportions[
                i - 1] + types_of_people[i].proportion
        return cumulative_proportions

    def get_type_of_person(self, cumulative_proportions, total_sum_of_proportions):
        prop_value = np.random.randint(0, total_sum_of_proportions)
        type_of_person = bisect.bisect(cumulative_proportions, prop_value)
        return type_of_person

    def get_prob_of_person(self, person_type_obj):
        return np.random.uniform(person_type_obj.low_prob, person_type_obj.high_prob)

    def get_correct_answer(self, ground_truths, question):
        answers = ground_truths[question]
        if self.answers_per_q == "single":
            return np.where(answers == 1)[0]
        correct_answer = np.zeros((self.options_per_q))
        for index in np.where(answers == 1):
            correct_answer[index] = 1
        return correct_answer

    def get_answer_of_person(self, correct_answer, prob_of_person, options_per_question):
        calc_prob = np.random.uniform(0, 1)
        person_answers_correctly = False
        if calc_prob <= prob_of_person:  # means his answer is correct
            person_answers_correctly = True
        if person_answers_correctly:
            return correct_answer
        else:
            answer = np.random.randint(0, options_per_question)
            while answer == correct_answer:
                answer = np.random.randint(0, options_per_question)
            return answer


class PeopleTypes:

    def __init__(self, proportion, low, high):
        assert(low >= 0 and low <= 1)
        assert(high >= 0 and high <= 1)
        self.proportion = proportion
        self.low_prob = low
        self.high_prob = high

if __name__ == "__main__":
    print("Simulator for simulating crowdsourced answers to multiple choice questions with single or multiple correct answers per question")
